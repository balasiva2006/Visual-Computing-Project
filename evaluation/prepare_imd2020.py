"""Create explicit external-test manifests without changing dataset files.

Run from the repository root: python evaluation/prepare_imd2020.py
Only enumerates paths; pixel and mask validation happens in the notebook.
"""
import argparse
import csv
import random
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def prepare(root, output, seed=42, preview_groups=5):
    root, output = Path(root).resolve(), Path(output)
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows, errors = [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        originals = [p for p in images if p.stem.lower().endswith('_orig')]
        masks = [p for p in images if p.stem.lower().endswith('_mask')]
        used = set()
        if len(originals) != 1:
            errors.append(f'{folder.name}: expected exactly one *_orig image, got {len(originals)}')
        for image in images:
            if image in masks:
                continue
            authentic = image in originals
            paired = [p for p in masks if p.stem == image.stem + '_mask']
            if not authentic and len(paired) != 1:
                errors.append(f'{image}: expected exactly one matching *_mask, got {len(paired)}')
                continue
            if not authentic:
                used.add(paired[0])
            rows.append(dict(id=image.relative_to(root).as_posix(),
                             image=image.relative_to(root).as_posix(),
                             mask='' if authentic else paired[0].relative_to(root).as_posix(),
                             label=int(not authentic), group=folder.name))
        errors.extend(f'Orphan mask: {p}' for p in masks if p not in used)
    if errors:
        raise ValueError('\n'.join(errors))
    if not rows:
        raise ValueError('No IMD2020 samples found')
    groups = sorted({row['group'] for row in rows})
    selected = set(random.Random(seed).sample(groups, min(preview_groups, len(groups))))
    output.mkdir(parents=True, exist_ok=True)
    for name, subset in [('imd2020_test.csv', rows),
                         ('imd2020_preview.csv', [r for r in rows if r['group'] in selected])]:
        with (output / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'image', 'mask', 'label', 'group'])
            writer.writeheader()
            writer.writerows(subset)
    return dict(images=len(rows), authentic=sum(r['label'] == 0 for r in rows),
                forged=sum(r['label'] == 1 for r in rows), groups=len(groups),
                preview_groups=len(selected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='datasets/imd2020')
    parser.add_argument('--out', default='evaluation/manifests')
    args = parser.parse_args()
    print(prepare(args.root, args.out))
