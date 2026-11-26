"""Build COCOGlide test manifests using the dataset's authoritative table.csv.

Only reads filenames and CSV metadata; pixel validation occurs in the notebook.
No images are copied, resized, or synthesized.
"""
import argparse
import csv
import random
import re
from pathlib import Path, PureWindowsPath

FIELDS = ['id', 'image', 'mask', 'label', 'group', 'source_image', 'prompt']
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def prepare(root, output, seed=42, preview_groups=5):
    root, output = Path(root).resolve(), Path(output)
    table = root / 'table.csv'
    if not table.is_file():
        raise FileNotFoundError(f'Missing {table}. Set DATA_ROOT to the extracted folder containing table.csv, real/, fake/, mask/.')

    def relative(value):
        path = PureWindowsPath((value or '').strip())
        if path.anchor or '..' in path.parts or not path.parts:
            raise ValueError(f'Expected a nonempty dataset-relative path: {value!r}')
        name = path.as_posix()
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f'Missing file or path outside dataset root: {value!r}')
        return name

    with table.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'real', 'fake', 'mask', 'prompt'} <= set(reader.fieldnames or []):
            raise ValueError('table.csv requires real,fake,mask,prompt columns.')
        pairs = list(reader)
    if not pairs:
        raise ValueError('Empty COCOGlide pairing table.')
    samples, used_fake, used_mask, used_real = {}, set(), set(), set()
    for line, pair in enumerate(pairs, 2):
        real, fake, mask = [relative(pair[key]) for key in ['real', 'fake', 'mask']]
        if real == fake or real == mask or fake == mask:
            raise ValueError(f'Overlapping image/mask roles at table line {line}.')
        if fake in used_fake or mask in used_mask:
            raise ValueError(f'Duplicate fake or reused mask at table line {line}.')
        # Metadata filenames carry the original MS-COCO ID; keep all crops/edits of it together.
        match = re.search(r'_(\d+)$', Path(real).stem)
        if not match:
            raise ValueError(f'Cannot recover source COCO ID from {real!r}; inspect table.csv before grouping.')
        group = 'coco_' + str(int(match.group(1)))
        prompt = (pair['prompt'] or '').strip()
        authentic = dict(id=real, image=real, mask='', label=0, group=group, source_image=real, prompt=prompt)
        if real in samples and samples[real] != authentic:
            raise ValueError(f'Conflicting source metadata for {real}.')
        if fake in samples or (real in samples and samples[real]['label'] != 0):
            raise ValueError('A file has conflicting image roles in table.csv.')
        samples[real] = authentic
        samples[fake] = dict(id=fake, image=fake, mask=mask, label=1, group=group, source_image=real, prompt=prompt)
        used_fake.add(fake); used_mask.add(mask); used_real.add(real)
    for folder, listed in [('real', used_real), ('fake', used_fake), ('mask', used_mask)]:
        actual = {p.relative_to(root).as_posix() for p in (root / folder).rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS}
        if actual != listed:
            raise ValueError(f'{folder}/ and table.csv disagree; unlisted={sorted(actual-listed)[:5]}, missing={sorted(listed-actual)[:5]}')
    rows = sorted(samples.values(), key=lambda row: row['id'])
    groups = sorted({r['group'] for r in rows})
    if preview_groups <= 0:
        raise ValueError('preview_groups must be positive.')
    chosen = set(random.Random(seed).sample(groups, min(preview_groups, len(groups))))
    output.mkdir(parents=True, exist_ok=True)
    for filename, subset in [('cocoglide_test.csv', rows),
                             ('cocoglide_preview.csv', [r for r in rows if r['group'] in chosen])]:
        with (output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(subset)
    return dict(table_pairs=len(pairs), images=len(rows), authentic=len(used_real), forged=len(used_fake),
                source_groups=len(groups), preview_images=sum(r['group'] in chosen for r in rows),
                preview_groups=len(chosen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='datasets/cocoglide')
    parser.add_argument('--out', default='evaluation/manifests')
    args = parser.parse_args()
    print(prepare(args.root, args.out))
