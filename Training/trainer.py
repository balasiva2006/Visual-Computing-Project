"""Stage-wise training. Run --smoke before launching a full stage."""
import argparse
import hashlib
import json
import math
import random
import tempfile
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data.dataset import CASIADataset
from data.frequency import ModelInputs, RGB_MEAN, RGB_STD, load_or_fit_stats
from data.transforms import train_transform, validation_transform
from models.model import ForgeryModel, INPUTS
from .checkpoints import restore_checkpoint, save_checkpoint, transfer_encoders
from .distillation import FrozenTeachers, distillation_loss
from .losses import classification_weight, forgery_loss


# mode, epoch cap, LR warmup epochs, earliest stopping epoch, patience
STAGES = {
    "spatial": ("spatial_only", 30, 3, 8, 6),
    "frequency": ("frequency_only", 50, 3, 8, 6),
    "warmup": ("fusion", 5, 0, 3, 1),
    "joint": ("fusion", 15, 1, 5, 4),
    "teacher": ("fusion", 15, 1, 5, 4),
    "direct": ("fusion", 30, 3, 8, 6),
}
VIEWS = ("clean", "jpeg70", "scale05", "blur1", "noise5")


class Inputs:
    def __init__(self, mode, view, stats):
        self.mode = mode
        self.transform = view if mode == "spatial_only" else ModelInputs(view, stats)

    def __call__(self, sample):
        s = self.transform(sample)
        if self.mode == "spatial_only":
            s["rgb"] = ((s["rgb"].astype(np.float32) / 255 - RGB_MEAN) / RGB_STD) * s["valid"]
        return {k: s[k] for k in (*INPUTS[self.mode], "mask")}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_identity(args):
    hashes = {s: digest(Path(args.root) / "csv files" / f"{s}.csv") for s in ("train", "val")}
    identity = {"csv": hashes, "stats": digest(args.stats)}
    if not args.smoke:
        audit = json.loads(Path(args.audit).read_text())
        if (audit["csv_sha256"] != hashes or audit["source_policy"] != "keep_existing"
                or audit["mask_foreground"] != "white"):
            raise ValueError("Audit and active CSVs differ. Audit the current dataset_clean first.")
        identity["pixels"] = audit["pixel_audit"]["retained_rgb_and_masks_sha256"]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def examples(ds):
    return [next(i for i, r in enumerate(ds.rows) if r["label"] == c) for c in (0, 1)]


def train_loader(ds, args, epoch):
    gen = torch.Generator().manual_seed(args.seed + epoch)
    if args.smoke:
        return DataLoader(Subset(ds, examples(ds)), batch_size=1)
    labels = torch.tensor([r["label"] for r in ds.rows])
    weights = 1.0 / torch.bincount(labels).double()[labels]
    sampler = WeightedRandomSampler(weights, len(ds), replacement=True, generator=gen)
    return DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                      generator=torch.Generator().manual_seed(args.seed + epoch + 10000))


def init_model(args, mode, data_id, device):
    use_imagenet = args.stage in ("spatial", "direct") and not (args.resume or args.smoke)
    model = ForgeryModel(mode, pretrained_spatial=use_imagenet)
    teachers = None
    if args.stage == "warmup":
        if not args.resume:
            loaded = transfer_encoders(model, args.spatial_ckpt, args.frequency_ckpt, data_id=data_id)
            (Path(args.out) / "transferred.json").write_text(json.dumps(loaded, indent=2))
        for enc in (model.spatial_encoder, model.frequency_encoder):
            enc.requires_grad_(False)
    elif args.stage in ("joint", "teacher"):
        warm = restore_checkpoint(args.warmup_ckpt, model, data_id=data_id, stage="warmup")
        if args.stage == "teacher":
            refs = []
            for stream, path in (("spatial", args.spatial_ckpt), ("frequency", args.frequency_ckpt)):
                if digest(path) != warm["config"]["parents"][stream]:
                    raise ValueError(f"{stream} teacher must be the checkpoint used for warmup")
                ref = ForgeryModel(stream + "_only", pretrained_spatial=False)
                restore_checkpoint(path, ref, data_id=data_id, stage=stream)
                refs.append(ref)
            teachers = FrozenTeachers(*refs).to(device)
    return model.to(device), teachers


def make_optimizer(model, stage):
    if stage in ("joint", "teacher"):
        slow = list(model.spatial_encoder.parameters()) + list(model.frequency_encoder.parameters())
        slow_lr, fast_lr = 1e-5, 1e-4
    elif stage in ("spatial", "direct"):
        slow = list(model.spatial_encoder.backbone.parameters())
        slow_lr, fast_lr = 1e-4, 3e-4
    else:
        slow, slow_lr, fast_lr = [], 0., 3e-4
    slow = [p for p in slow if p.requires_grad]
    slow_ids = {id(p) for p in slow}
    fast = [p for p in model.parameters() if p.requires_grad and id(p) not in slow_ids]
    groups = [{"params": fast, "lr": fast_lr}]
    if slow:
        groups.append({"params": slow, "lr": slow_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def lr_factor(index, cap, warm):
    if warm == 0:
        return 1.0
    if index < warm:
        return (index + 1) / warm
    phase = min(1., (index + 1 - warm) / max(cap - warm, 1))
    return .1 + .45 * (1 + math.cos(math.pi * phase))


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def train_epoch(model, teachers, loader, opt, scaler, args, epoch, device):
    model.train()
    if args.stage == "warmup":
        model.spatial_encoder.eval()
        model.frequency_encoder.eval()
    lam = classification_weight(epoch, standalone=args.stage in ("spatial", "frequency"))
    steps = 2 if args.smoke else 8 // args.batch_size
    iterator = iter(loader)
    total, images, updates = 0., 0, 0
    while True:
        window = list(islice(iterator, steps))
        if not window:
            break
        n = sum(b["label"].numel() for b in window)
        positives = sum(int(b["label"].sum()) for b in window)
        opt.zero_grad(set_to_none=True)
        for cpu in window:
            batch = move(cpu, device)
            fraction = batch["label"].numel() / n
            pos_fraction = int(cpu["label"].sum()) / max(1, positives)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                out = model(batch, return_features=teachers is not None)
                loss = forgery_loss(out, batch, lambda_cls=lam)
                # Weight Dice by positive examples across the accumulation window.
                objective = fraction * (.5 * loss["pixel_bce"] + lam * loss["image_bce"])
                objective = objective + .5 * pos_fraction * loss["dice"]
                if teachers is not None:
                    dist = distillation_loss(out["features"], teachers(batch), batch["valid"])
                    objective = objective + fraction * args.beta * dist
            if not torch.isfinite(objective).item():
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
            scaler.scale(objective).backward()
            total += objective.detach().item() * n
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        updates += int(scaler.get_scale() >= scale)
        images += n
    if not updates:
        raise RuntimeError("No optimizer updates completed")
    return total / images, updates


@torch.no_grad()
def validate(model, loaders, device):
    model.eval()
    results = {}
    for view, loader in loaders.items():
        labels, scores, dice, pristine_area = [], [], [], []
        for cpu in loader:  # One native-resolution image at a time.
            b = move(cpu, device)
            out = model(b)
            logit = out["image_logits"].item()
            if not math.isfinite(logit) or not out["mask_logits"].isfinite().all().item():
                raise FloatingPointError(f"Nonfinite validation output: {view}")
            pred = (out["mask_logits"] >= 0) & b["valid"]
            target = (b["mask"] > .5) & b["valid"]
            label = int(b["label"].item())
            labels.append(label)
            scores.append(logit)
            if label:
                dice.append((2 * (pred & target).sum() / (pred.sum() + target.sum())).item())
            else:
                pristine_area.append((pred.sum() / b["valid"].sum()).item())
        pos = np.asarray(labels, dtype=bool)
        npos, nneg = int(pos.sum()), int((~pos).sum())
        if min(npos, nneg) == 0:
            raise ValueError("Validation requires both image classes")
        auc = (rankdata(scores)[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        results[view] = dict(dice=float(np.mean(dice)), auc=float(auc),
                             pristine_area=float(np.mean(pristine_area)),
                             image_fpr=float((np.asarray(scores)[~pos] >= 0).mean()))
    score = float(np.mean([.5 * (r["dice"] + r["auc"]) for r in results.values()]))
    return score, results


def run(args):
    mode, cap, warm, minimum, patience = STAGES[args.stage]
    cap = args.epochs or cap
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use a GPU allocation or --device cpu.")
    output = Path(args.out)
    last, best = output / "last.pt", output / "best.pt"
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError(f"{output} is nonempty. Choose a new --out or use --resume.")
    if args.resume and not (last.is_file() and best.is_file()):
        raise FileNotFoundError("Resume needs both last.pt and best.pt in the same run directory")

    seed_all(args.seed)
    stats = load_or_fit_stats(args.root, args.stats)
    data_id = data_identity(args)
    parents = {k: digest(v) for k, v in (("spatial", args.spatial_ckpt),
               ("frequency", args.frequency_ckpt), ("warmup", args.warmup_ckpt)) if v}
    config = dict(stage=args.stage, epochs=cap, seed=args.seed, microbatch=args.batch_size,
                  crop=args.crop, workers=args.workers, amp=args.amp, beta=args.beta,
                  parents=parents, smoke=args.smoke)
    model, teachers = init_model(args, mode, data_id, device)
    if args.spatial_last_module:
        backbone = model.spatial_encoder.backbone
        backbone.requires_grad_(False)
        backbone.stage4[-1].requires_grad_(True)
        config["spatial_last_module"] = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        spatial_trainable = sum(
            p.numel() for p in model.spatial_encoder.parameters() if p.requires_grad
        )
        print(f"Spatial partial fine-tuning: {spatial_trainable:,} spatial trainable parameters; "
              f"{trainable:,} total trainable parameters", flush=True)
    opt = make_optimizer(model, args.stage)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lr_factor(i, cap, warm))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    train = CASIADataset(args.root, "train", transform=Inputs(mode, partial(train_transform, crop_size=args.crop), stats))
    loaders = {}
    for view in VIEWS:
        ds = CASIADataset(args.root, "val", transform=Inputs(mode, partial(validation_transform, view=view, seed=42), stats))
        val = Subset(ds, examples(ds)) if args.smoke else ds
        loaders[view] = DataLoader(val, batch_size=1, num_workers=args.workers)

    progress = {"best": -1., "reference": -1., "stale": 0, "updates": 0, "history": []}
    start = 1

    def save(path, epoch):
        save_checkpoint(path, model, stage=args.stage, epoch=epoch, data_id=data_id, config=config,
                        progress=progress, optimizer=opt, scheduler=sched, scaler=scaler)

    if args.resume:
        meta = restore_checkpoint(last, model, data_id=data_id, stage=args.stage,
                                  optimizer=opt, scheduler=sched, scaler=scaler)
        if config != meta["config"]:
            raise ValueError("Run configuration differs from the saved checkpoint")
        start, progress = meta["epoch"] + 1, meta["progress"]
        if start > minimum and progress["stale"] >= patience:
            print("Run already stopped by early stopping.")
            return
    elif args.stage in ("joint", "teacher"):
        # Epoch zero is the warmup candidate; preserve it if joint tuning gets worse.
        score, metrics = validate(model, loaders, device)
        progress.update(best=score, reference=score)
        progress["history"].append(dict(epoch=0, score=score, validation=metrics))
        save(best, 0)

    print(f"{args.stage}: device={device}, epochs={cap}, training rows={len(train)}", flush=True)
    for epoch in range(start, cap + 1):
        seed_all(args.seed + epoch)
        if args.smoke:
            before = model.heads.mask_head.weight.detach().clone()
        value, updates = train_epoch(model, teachers, train_loader(train, args, epoch), opt, scaler, args, epoch, device)
        if args.smoke and torch.equal(before, model.heads.mask_head.weight):
            raise RuntimeError("Smoke test: segmentation head did not update")
        score, metrics = validate(model, loaders, device)
        lrs = [g["lr"] for g in opt.param_groups]
        sched.step()
        record = dict(epoch=epoch, train_loss=value, updates=updates, lr=lrs, score=score, validation=metrics)
        progress["updates"] += updates
        progress["history"].append(record)
        if score > progress["reference"] + .001:
            progress.update(reference=score, stale=0)
        else:
            progress["stale"] += 1
        if score > progress["best"]:
            progress["best"] = score
            save(best, epoch)
        save(last, epoch)
        print(json.dumps(record), flush=True)
        if epoch >= minimum and progress["stale"] >= patience:
            break
    if args.smoke:
        restore_checkpoint(best, model, data_id=data_id, stage=args.stage)
        print("Trainer smoke check passed: update, five validation views, checkpoint reload")
    else:
        print(f"Best score: {progress['best']:.5f} | {best}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Optional YAML mapping using the CLI option names with underscores")
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--root", default="./dataset_clean")
    p.add_argument("--stats", default="./artifacts/dct_stats.json")
    p.add_argument("--audit", default="./artifacts/casia_clean_split.json")
    p.add_argument("--out")
    p.add_argument("--spatial-ckpt")
    p.add_argument("--frequency-ckpt")
    p.add_argument("--warmup-ckpt")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--beta", type=float, default=.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--spatial-last-module", action="store_true",
                   help="During spatial or joint training, freeze HRNet except its final stage-4 module; keep spatial adapters trainable")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    initial, _ = p.parse_known_args()
    if initial.config:
        import yaml
        settings = yaml.safe_load(Path(initial.config).read_text())
        if not isinstance(settings, dict) or set(settings) - vars(initial).keys():
            p.error("YAML must be a mapping of known trainer options")
        p.set_defaults(**settings)
    args = p.parse_args()
    if args.spatial_last_module and (args.stage not in ("spatial", "joint") or args.smoke):
        p.error("--spatial-last-module requires the spatial or joint training stage")
    if args.smoke:
        args.stage, args.crop, args.batch_size, args.epochs, args.workers = "frequency", 128, 1, 1, 0
        args.resume, args.amp = False, False
        with tempfile.TemporaryDirectory() as tmp:
            args.out = tmp
            run(args)
        return
    if args.stage not in STAGES or not args.out:
        p.error("Choose --smoke, or specify --stage and --out")
    if args.batch_size not in (1, 2, 4, 8) or args.crop <= 0 or args.crop % 32:
        p.error("Choose a microbatch of 1, 2, 4 or 8 and a positive crop divisible by 32")
    if args.workers < 0 or (args.epochs is not None and args.epochs < 1) or args.beta < 0:
        p.error("Invalid workers, epoch cap or beta")
    if args.stage in ("warmup", "teacher") and not (args.spatial_ckpt and args.frequency_ckpt):
        p.error("Supply both --spatial-ckpt and --frequency-ckpt")
    if args.stage in ("joint", "teacher") and not args.warmup_ckpt:
        p.error("Supply --warmup-ckpt pointing to the best fusion warmup checkpoint")
    if args.stage == "teacher" and args.beta <= 0:
        p.error("Use the joint stage for the beta=0 comparison")
    run(args)


if __name__ == "__main__":
    main()
