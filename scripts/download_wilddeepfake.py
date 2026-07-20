"""
scripts/download_wilddeepfake.py
================================
Download WildDeepfake dataset from HuggingFace and prepare
train/test frame structure for MSTF-Net.

Dataset: xingjunm/WildDeepfake
  - ~72GB raw tar.gz files
  - Already extracted PNG face crops (no video processing)
  - Official train/test split
  - ~3805 real + ~3509 fake sequences

After this script:
    wilddeepfake/
    ├── train/
    │   ├── real/  {seq_id}/  *.png
    │   └── fake/  {seq_id}/  *.png
    └── test/
        ├── real/  {seq_id}/  *.png
        └── fake/  {seq_id}/  *.png

Then zip frames → upload to Drive → restore each session in ~5 min.

Usage:
    python scripts/download_wilddeepfake.py \
        --output_dir /teamspace/studios/this_studio/wilddeepfake \
        --cache_dir  /teamspace/studios/this_studio/hf_cache

Time estimate on Lightning A100/H100:
    Download  : ~60-90 min (72GB at ~15MB/s HuggingFace speed)
    Extract   : ~20 min
    Total     : ~90-110 min
"""

import argparse
import os
import shutil
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir',
                   default='/teamspace/studios/this_studio/wilddeepfake')
    p.add_argument('--cache_dir',
                   default='/teamspace/studios/this_studio/hf_cache')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--skip_download', action='store_true',
                   help='Skip download if already done')
    return p.parse_args()


def download_dataset(cache_dir: str) -> str:
    """Download WildDeepfake from HuggingFace."""
    from huggingface_hub import snapshot_download
    print('Downloading WildDeepfake from HuggingFace...')
    print('  Repo : xingjunm/WildDeepfake')
    print('  Size : ~72GB')
    print('  Time : ~60-90 min')
    print()

    path = snapshot_download(
        repo_id='xingjunm/WildDeepfake',
        repo_type='dataset',
        local_dir=cache_dir,
        ignore_patterns=['*.parquet', '*.arrow', 'data/'],
        force_download=True,   # don't trust a possibly-corrupt local cache
    )
    print(f'✅ Downloaded to: {path}')

    # Integrity check: verify files are openable tar archives (gzip OR
    # plain — HF serves these as plain uncompressed tar despite the
    # .tar.gz name), and catch genuine corruption/truncation.
    bad = []
    checked = 0
    for tar_path in Path(path).rglob('*.tar.gz'):
        checked += 1
        with open(tar_path, 'rb') as f:
            magic = f.read(262)
        is_gzip = magic[:2] == b'\x1f\x8b'
        is_tar = magic[257:262] in (b'ustar', b'ustar\x00'[:5])
        if not (is_gzip or is_tar):
            bad.append(tar_path)
    if bad:
        print(f'❌ {len(bad)}/{checked} files are neither valid gzip nor '
              f'valid tar (genuinely corrupt/truncated).')
        print(f'   First bad file: {bad[0]}')
        raise RuntimeError(
            'Downloaded archive is corrupted. Delete cache_dir and '
            '~/.cache/huggingface/hub/datasets--xingjunm--WildDeepfake '
            'and rerun (check disk space with df -h first).'
        )
    print(f'✅ Verified {checked} archives (plain tar, not gzip — extraction uses r:* to auto-detect)')
    return path


def extract_tar(tar_path: Path, out_dir: Path) -> bool:
    """Extract a single tar file. Despite the .tar.gz extension, HuggingFace
    serves these as plain uncompressed POSIX tar (confirmed via `file`).
    Use 'r:*' to auto-detect gzip vs plain tar instead of forcing gzip."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(tar_path), 'r:*') as tf:
            tf.extractall(str(out_dir))
        return True
    except Exception as e:
        print(f'  ❌ Failed {tar_path.name}: {e}')
        return False


def prepare_structure(cache_dir: str, output_dir: str, workers: int):
    """
    Extract tar.gz files and organise into train/test structure.

    WildDeepfake raw structure:
        deepfake_in_the_wild/
        ├── fake_train/ *.tar.gz  → output/train/fake/
        ├── real_train/ *.tar.gz  → output/train/real/
        ├── fake_test/  *.tar.gz  → output/test/fake/
        └── real_test/  *.tar.gz  → output/test/real/
    """
    cache_dir  = Path(cache_dir)
    output_dir = Path(output_dir)

    # Map source folder → destination
    mapping = {
        'fake_train' : output_dir / 'train' / 'fake',
        'real_train' : output_dir / 'train' / 'real',
        'fake_test'  : output_dir / 'test'  / 'fake',
        'real_test'  : output_dir / 'test'  / 'real',
    }

    # Find the deepfake_in_the_wild folder
    raw_root = None
    for candidate in [
        cache_dir / 'deepfake_in_the_wild',
        cache_dir / 'data' / 'deepfake_in_the_wild',
        cache_dir,
    ]:
        if candidate.exists():
            # Check if it has the expected folders
            if any((candidate / k).exists() for k in mapping.keys()):
                raw_root = candidate
                break

    if raw_root is None:
        # Search recursively
        for p in cache_dir.rglob('fake_train'):
            raw_root = p.parent
            break

    if raw_root is None:
        print(f'❌ Could not find WildDeepfake structure in {cache_dir}')
        print(f'   Contents: {list(cache_dir.iterdir())[:10]}')
        raise FileNotFoundError('WildDeepfake raw structure not found')

    print(f'Found raw data at: {raw_root}')

    total_extracted = 0
    for folder_name, dest_dir in mapping.items():
        src_dir = raw_root / folder_name
        if not src_dir.exists():
            print(f'  ⚠️  {folder_name} not found — skipping')
            continue

        tar_files = sorted(src_dir.glob('*.tar.gz'))
        if not tar_files:
            # Maybe already extracted
            subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
            if subdirs:
                print(f'  {folder_name}: already extracted ({len(subdirs)} sequences)')
                # Move/copy to dest
                dest_dir.mkdir(parents=True, exist_ok=True)
                for sd in subdirs:
                    target = dest_dir / sd.name
                    if not target.exists():
                        shutil.move(str(sd), str(target))
                total_extracted += len(subdirs)
            continue

        print(f'\n  Extracting {folder_name} ({len(tar_files)} tar.gz files)...')
        dest_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(extract_tar, tf, dest_dir): tf
                for tf in tar_files
            }
            for f in as_completed(futs):
                if f.result():
                    done += 1
                if done % 50 == 0:
                    print(f'    {done}/{len(tar_files)} extracted')

        total_extracted += done
        print(f'  ✅ {folder_name}: {done} sequences extracted')

    return total_extracted


def verify(output_dir: str):
    """Count and verify frame structure."""
    output_dir = Path(output_dir)
    print('\nVerifying structure...')

    counts = {}
    for split in ['train', 'test']:
        for label in ['real', 'fake']:
            d = output_dir / split / label
            if d.exists():
                seqs   = sum(1 for x in d.iterdir() if x.is_dir())
                frames = sum(1 for _ in d.rglob('*.png'))
                frames += sum(1 for _ in d.rglob('*.jpg'))
                counts[f'{split}/{label}'] = {'sequences': seqs, 'frames': frames}
                print(f'  {split}/{label}: {seqs} sequences, {frames:,} frames')
            else:
                print(f'  ⚠️  {split}/{label} not found')

    total_frames = sum(v['frames'] for v in counts.values())
    print(f'\n  Total frames: {total_frames:,}')

    if total_frames < 10000:
        print('❌ Too few frames — something went wrong')
        return False

    print('✅ Structure looks good')
    return True


def zip_and_upload(output_dir: str):
    """Zip frames and upload to Drive for persistence."""
    print('\nZipping frames for Drive persistence...')
    zip_base = '/teamspace/studios/this_studio/wilddeepfake_frames'
    zip_path = zip_base + '.zip'

    import subprocess
    # Use stdlib instead of shelling out to `zip` — that binary isn't
    # installed on all Lightning images.
    try:
        shutil.make_archive(zip_base, 'zip', root_dir=output_dir)
        zip_ok = True
    except Exception as e:
        print(f'❌ Zip failed: {e}')
        zip_ok = False

    if zip_ok:
        size = Path(zip_path).stat().st_size / 1e9
        print(f'✅ Zipped: {zip_path} ({size:.1f}GB)')

        print('Uploading to Drive...')
        result2 = subprocess.run(
            ['rclone', 'copy', zip_path, 'gdrive:',
             '--progress', '--transfers', '4'],
            capture_output=False
        )
        if result2.returncode == 0:
            print('✅ Uploaded to gdrive:wilddeepfake_frames.zip')
            os.remove(zip_path)
            print('   Local zip deleted')
        else:
            print(f'⚠️  Upload failed — zip saved at {zip_path}')


def main():
    args = parse_args()

    print('=' * 60)
    print('WildDeepfake — Download & Prepare')
    print(f'Output : {args.output_dir}')
    print(f'Cache  : {args.cache_dir}')
    print('=' * 60)

    # Check if already prepared
    output_dir = Path(args.output_dir)
    existing = sum(1 for _ in output_dir.rglob('*.png')) if output_dir.exists() else 0
    if existing > 10000:
        print(f'✅ Already prepared ({existing:,} frames)')
        verify(args.output_dir)
        zip_and_upload(args.output_dir)
        return

    # Download
    if not args.skip_download:
        cache_path = download_dataset(args.cache_dir)
    else:
        cache_path = args.cache_dir
        print(f'Skipping download — using {cache_path}')

    # Extract and organise
    print('\nExtracting and organising...')
    n = prepare_structure(cache_path, args.output_dir, args.workers)
    print(f'✅ Extracted {n} sequences')

    # Verify
    ok = verify(args.output_dir)
    if not ok:
        print('❌ Verification failed')
        return

    # Zip and upload to Drive
    zip_and_upload(args.output_dir)

    print('\n' + '=' * 60)
    print('DATASET READY ✅')
    print(f'  Frames at : {args.output_dir}')
    print(f'  Drive     : gdrive:wilddeepfake_frames.zip')
    print()
    print('Next:')
    print('  bash lightning/train_session.sh --session 1')
    print('=' * 60)


if __name__ == '__main__':
    main()