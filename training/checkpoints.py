"""Save a training run, resume its stage, or transfer standalone encoders."""
import os
import tempfile
from pathlib import Path

import torch


def save_checkpoint(path, model, *, stage, epoch, data_id, config, progress,
                    optimizer=None, scheduler=None, scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "version": 1,
        "mode": model.mode,
        "stage": stage,
        "epoch": epoch,
        "data_id": data_id,
        "config": config,
        "progress": progress,
        "model": model.state_dict(),
    }
    for name, obj in (
        ("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)
    ):
        if obj is not None:
            state[name] = obj.state_dict()

    # Finish writing the new checkpoint before replacing the previous file.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _read(path, mode, data_id):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != 1 or state["mode"] != mode:
        raise ValueError(f"{path}: incompatible format or model mode; expected {mode}")
    if state["data_id"] != data_id:
        raise ValueError(f"{path}: different training data or preprocessing")
    return state


def restore_checkpoint(path, model, *, data_id, stage=None,
                       optimizer=None, scheduler=None, scaler=None):
    state = _read(path, model.mode, data_id)
    objects = (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler))

    if any(obj is not None for _, obj in objects) and stage is None:
        raise ValueError("Specify the original stage to resume optimizer/scheduler/scaler state")
    if stage is not None and state["stage"] != stage:
        raise ValueError(f"Expected stage {stage}, found {state['stage']}")
    for name, obj in objects:
        if obj is not None and name not in state:
            raise ValueError(f"{path}: no saved {name} state")

    model.load_state_dict(state["model"], strict=True)
    for name, obj in objects:
        if obj is not None:
            obj.load_state_dict(state[name])

    return {key: state[key] for key in ("epoch", "stage", "config", "progress")}


def transfer_encoders(fusion, spatial_path, frequency_path, *, data_id):
    if fusion.mode != "fusion":
        raise ValueError("The destination must be a fusion model")

    # Check both encoders before copying either one.
    transfers = []
    for stream, path in (("spatial", spatial_path), ("frequency", frequency_path)):
        state = _read(path, stream + "_only", data_id)
        prefix = stream + "_encoder."
        weights = {
            k[len(prefix):]: v
            for k, v in state["model"].items()
            if k.startswith(prefix)
        }

        encoder = getattr(fusion, stream + "_encoder")
        expected = encoder.state_dict()

        if weights.keys() != expected.keys():
            missing = sorted(expected.keys() - weights.keys())
            extra = sorted(weights.keys() - expected.keys())
            raise ValueError(f"{stream}: missing={missing}, unexpected={extra}")

        for name, value in weights.items():
            if value.shape != expected[name].shape:
                raise ValueError(f"{stream}: incompatible shape for {name}")

        transfers.append((stream, encoder, weights))

    loaded = {}
    for stream, encoder, weights in transfers:
        encoder.load_state_dict(weights, strict=True)
        loaded[stream] = sorted(weights)

    return loaded


def main():
    # Test on tiny models with the same encoder names used by ForgeryModel.
    from torch import nn

    class Toy(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            if mode != "frequency_only":
                self.spatial_encoder = nn.Linear(2, 2)
            if mode != "spatial_only":
                self.frequency_encoder = nn.Linear(2, 2)
            self.heads = nn.Linear(2, 1)

    def update(model, optimizer):
        optimizer.zero_grad()
        model.heads(model.spatial_encoder(torch.ones(1, 2))).sum().backward()
        optimizer.step()

    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        spatial = Toy("spatial_only")
        frequency = Toy("frequency_only")
        fusion = Toy("fusion")

        opt = torch.optim.AdamW(spatial.parameters(), lr=1e-3)
        update(spatial, opt)

        args = dict(
            stage="standalone", epoch=1, data_id="toy",
            config={}, progress={"best": 0.7},
        )
        save_checkpoint(root / "s.pt", spatial, optimizer=opt, **args)
        save_checkpoint(root / "f.pt", frequency, **args)

        # Both encoders must match their source. Fusion's own heads must
        # retain the values they had before the transfer.
        before = {k: v.clone() for k, v in fusion.heads.state_dict().items()}
        transfer_encoders(fusion, root / "s.pt", root / "f.pt", data_id="toy")

        for src, dst in (
            (spatial.spatial_encoder, fusion.spatial_encoder),
            (frequency.frequency_encoder, fusion.frequency_encoder),
        ):
            for k, v in src.state_dict().items():
                torch.testing.assert_close(v, dst.state_dict()[k])
        for k, v in before.items():
            torch.testing.assert_close(v, fusion.heads.state_dict()[k])

        # On the same next batch, a resumed AdamW optimizer should produce
        # the same update as the uninterrupted optimizer.
        resumed = Toy("spatial_only")
        opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        meta = restore_checkpoint(
            root / "s.pt", resumed, data_id="toy",
            stage="standalone", optimizer=opt2,
        )
        assert meta["epoch"] == 1 and meta["progress"]["best"] == 0.7

        update(spatial, opt)
        update(resumed, opt2)
        for k, v in spatial.state_dict().items():
            torch.testing.assert_close(v, resumed.state_dict()[k])

    print("Checkpoint check passed: resume and encoder-only transfer")


if __name__ == "__main__":
    main()
