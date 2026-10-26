"""Copy images referenced by the cleaned GQA annotation splits.

Example (PowerShell):
    python scripts/gqa_probing/copy_cleaned_gqa_images.py `
      --images-dir data/gqa/images `
      --cleaned-dir data/gqa_cleaned

Requires cleaned-dir/image_ids/train.txt and val.txt produced by
prepare_gqa_probing.py. Original image files are never modified.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def read_ids(cleaned_dir: Path, splits: list[str]) -> set[str]:
    ids = set()
    for split in splits:
        path = cleaned_dir / 'image_ids' / f'{split}.txt'
        if not path.is_file():
            raise FileNotFoundError(f'Missing image ID list: {path}')
        values = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        print(f'{split}: {len(values)} referenced image IDs')
        ids.update(values)
    return ids


def find_image(images_dir: Path, image_id: str) -> Path | None:
    # GQA images are normally stored as <image_id>.jpg in one directory.
    for ext in EXTENSIONS:
        candidate = images_dir / f'{image_id}{ext}'
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', required=True, type=Path, help='Original GQA images folder')
    parser.add_argument('--cleaned-dir', required=True, type=Path, help='Output folder of prepare_gqa_probing.py')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--dry-run', action='store_true', help='Check references without copying')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing copied images')
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    cleaned_dir = args.cleaned_dir.resolve()
    destination = cleaned_dir / 'images'
    if not images_dir.is_dir():
        parser.error(f'Original images directory not found: {images_dir}')
    if images_dir == destination or destination in images_dir.parents:
        parser.error('Destination must not be the original images folder or a parent of it')

    ids = read_ids(cleaned_dir, args.splits)
    if not args.dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    missing = []
    for image_id in sorted(ids):
        source = find_image(images_dir, image_id)
        if source is None:
            counts['missing'] += 1
            missing.append(image_id)
            continue
        target = destination / source.name
        if target.exists() and not args.overwrite:
            counts['already_present'] += 1
            continue
        if not args.dry_run:
            shutil.copy2(source, target)
        counts['would_copy' if args.dry_run else 'copied'] += 1

    report = {
        'source_images_dir': str(images_dir),
        'destination_images_dir': str(destination),
        'splits': args.splits,
        'unique_ids': len(ids),
        'dry_run': args.dry_run,
        'counts': dict(counts),
        'missing_ids': missing,
    }
    if not args.dry_run:
        report_path = cleaned_dir / 'image_copy_report.json'
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Report: {report_path}')
    print(f'Unique IDs: {len(ids)} | Copied: {counts["copied"]} | Existing: {counts["already_present"]} | Missing: {counts["missing"]}'
          + (f' | Would copy: {counts["would_copy"]}' if args.dry_run else ''))
    if missing:
        print('First missing IDs:', ', '.join(missing[:10]))


if __name__ == '__main__':
    main()
