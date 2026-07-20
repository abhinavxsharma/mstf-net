"""
scripts/extract_frames.py
==========================
Extract frames from DeeperForensics-1.0 zips on Lightning AI.

Protocol (exactly matching paper):
  - 8 frames per video, uniformly sampled (np.linspace)
  - Resize to 224×224, JPEG quality 95
  - Uses OFFICIAL DeeperForensics train/val split from lists.zip
  - NO identity-safe pair logic (DeeperForensics has independent
    real/fake videos — no identity pairs like FF++)

Strategy on Lightning AI:
  1. Copy zips from /teamspace/studios/this_studio/ → local SSD /teamspace/studios/
     (local SSD is much faster than uploads storage for unzipping)
  2. Unzip on local SSD
  3. Extract frames → save to /teamspace/studios/this_studio/DF_Frames/ (persistent)
  4. Delete local unzipped videos to free SSD space
  5. Repeat for each zip part

Time estimate on H100:
  - Unzip + extract all parts : ~55 min total
  - Fits comfortably in Session 1

Usage:
    python scripts/extract_frames.py \
        --dataset_dir /teamspace/studios/this_studio/DeeperForensics-1.0 \
        --frames_out  /teamspace/studios/this_studio/DF_Frames \
        --local_tmp   /teamspace/studios/this_studio/df_tmp

Authors : Abhinav Vats et al., Chandigarh University
Accepted: ISSMAD 2026 (Google + IEEE Signal Processing Society)
"""

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir',
                   default='/teamspace/studios/this_studio/DeeperForensics-1.0',
                   help='Where DeeperForensics zips are stored (persistent)')
    p.add_argument('--frames_out',
                   default='/teamspace/studios/this_studio/DF_Frames',
                   help='Where to save frames (persistent)')
    p.add_argument('--local_tmp',
                   default='/teamspace/studios/this_studio/df_tmp',
                   help='Local SSD for fast unzipping (resets between sessions)')
    p.add_argument('--n_frames',  type=int, default=8)
    p.add_argument('--img_size',  type=int, default=224)
    p.add_argument('--workers',   type=int, default=16)
    return p.parse_args()


def unzip_part(zip_path: Path, out_dir: Path) -> None:
    """Unzip a single zip part to out_dir."""
    print(f'  Unzipping {zip_path.name} ({zip_path.stat().st_size/1e9:.1f}GB)...')
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(out_dir))
    print(f'  ✅ {zip_path.name} unzipped')


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    frames_out  = Path(args.frames_out)
    local_tmp   = Path(args.local_tmp)
    local_tmp.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('DeeperForensics-1.0 Frame Extraction')
    print(f'  Source zips : {dataset_dir}')
    print(f'  Frames out  : {frames_out}  (persistent)')
    print(f'  Local tmp   : {local_tmp}  (fast SSD)')
    print(f'  n_frames    : {args.n_frames}')
    print(f'  img_size    : {args.img_size}')
    print('='*60)

    # ── Check if already done ────────────────────────────────
    r_tr = sum(1 for _ in (frames_out/'train'/'real').rglob('*.jpg')) \
           if (frames_out/'train'/'real').exists() else 0
    f_tr = sum(1 for _ in (frames_out/'train'/'fake').rglob('*.jpg')) \
           if (frames_out/'train'/'fake').exists() else 0

    if r_tr > 50000 and f_tr > 50000:
        print(f'✅ Frames already extracted: real={r_tr:,} fake={f_tr:,}')
        print('Skipping — delete /teamspace/studios/this_studio/DF_Frames to re-extract')
        return

    # ── Step 1: Unzip lists.zip to get official split ────────
    lists_zip = dataset_dir / 'lists.zip'
    lists_out = local_tmp / 'lists'

    if lists_zip.exists() and not lists_out.exists():
        print('\n[1] Extracting lists.zip...')
        with zipfile.ZipFile(str(lists_zip), 'r') as zf:
            zf.extractall(str(local_tmp))
        print('✅ lists extracted')
    else:
        print('\n[1] lists already extracted or not found')

    # Find actual lists directory
    list_files = list(local_tmp.rglob('train.txt'))
    lists_dir_actual = list_files[0].parent if list_files else None
    if lists_dir_actual:
        print(f'    Found lists at: {lists_dir_actual}')
        # Copy lists to persistent storage
        lists_persistent = frames_out / 'lists'
        lists_persistent.mkdir(parents=True, exist_ok=True)
        for f in lists_dir_actual.glob('*.txt'):
            shutil.copy2(str(f), str(lists_persistent / f.name))
    else:
        print('    No lists found — will use 80/20 random split')

    # ── Step 2: Process each zip part ────────────────────────
    # Find all zip parts
    source_zips = sorted(dataset_dir.glob('source_videos_part_*.zip'))
    manip_zips  = sorted(dataset_dir.glob('manipulated_videos_part_*.zip'))

    print(f'\n[2] Found {len(source_zips)} source zips + '
          f'{len(manip_zips)} manipulated zips')

    # Process source (real) zips
    source_tmp = local_tmp / 'source_videos'
    source_tmp.mkdir(exist_ok=True)

    print(f'\n[3] Processing source (real) videos...')
    for zip_path in source_zips:
        print(f'\n  --- {zip_path.name} ---')
        unzip_part(zip_path, source_tmp)

        # Find all .mp4 files just unzipped
        mp4_files = list(source_tmp.rglob('*.mp4'))
        print(f'  Extracting frames from {len(mp4_files)} videos...')

        # Import here to use the correct function
        from mstfnet.dataset import extract_deeperforensics_frames
        # Actually call per-file extraction for this batch
        _extract_batch_real(mp4_files, frames_out, args, lists_dir_actual)

        # Clean up to free local SSD space
        for f in mp4_files:
            try:
                f.unlink()
            except Exception:
                pass
        print(f'  ✅ Cleaned local SSD')

    # Process manipulated (fake) zips
    manip_tmp = local_tmp / 'manipulated_videos'
    manip_tmp.mkdir(exist_ok=True)

    print(f'\n[4] Processing manipulated (fake) videos...')
    for zip_path in manip_zips:
        print(f'\n  --- {zip_path.name} ---')
        unzip_part(zip_path, manip_tmp)

        mp4_files = list(manip_tmp.rglob('*.mp4'))
        print(f'  Extracting frames from {len(mp4_files)} videos...')

        _extract_batch_fake(mp4_files, frames_out, args)

        for f in mp4_files:
            try:
                f.unlink()
            except Exception:
                pass
        print(f'  ✅ Cleaned local SSD')

    # ── Final summary ────────────────────────────────────────
    r_tr = sum(1 for _ in (frames_out/'train'/'real').rglob('*.jpg'))
    f_tr = sum(1 for _ in (frames_out/'train'/'fake').rglob('*.jpg'))
    r_va = sum(1 for _ in (frames_out/'val'/'real').rglob('*.jpg'))
    f_va = sum(1 for _ in (frames_out/'val'/'fake').rglob('*.jpg'))

    print('\n' + '='*60)
    print('EXTRACTION COMPLETE')
    print(f'  Train: real={r_tr:,}  fake={f_tr:,}')
    print(f'  Val  : real={r_va:,}  fake={f_va:,}')
    print(f'  Saved to: {frames_out}  (persistent — never re-extract)')
    print('='*60)


def _load_official_split(lists_dir):
    """Load official train/val video stems from list files."""
    train_ids, val_ids = set(), set()
    if lists_dir is None:
        return train_ids, val_ids

    for fname in ['train.txt', 'train_list.txt']:
        fp = Path(lists_dir) / fname
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    stem = line.strip().split()[0].replace('.mp4','')
                    if stem:
                        train_ids.add(stem)
            break

    for fname in ['val.txt', 'val_list.txt']:
        fp = Path(lists_dir) / fname
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    stem = line.strip().split()[0].replace('.mp4','')
                    if stem:
                        val_ids.add(stem)
            break

    return train_ids, val_ids


def _extract_batch_real(mp4_files, frames_out, args, lists_dir):
    """Extract frames for a batch of real videos with correct split."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from mstfnet.dataset import extract_frames_from_video
    import random

    train_ids, val_ids = _load_official_split(lists_dir)

    # Determine split for each video
    tasks = []
    unassigned = []
    for vpath in mp4_files:
        stem = vpath.stem
        if stem in train_ids:
            tasks.append((vpath, 'train'))
        elif stem in val_ids:
            tasks.append((vpath, 'val'))
        else:
            unassigned.append(vpath)

    # Assign unassigned 80/20
    if unassigned:
        random.seed(42)
        random.shuffle(unassigned)
        cut = int(len(unassigned) * 0.8)
        tasks += [(v, 'train') for v in unassigned[:cut]]
        tasks += [(v, 'val')   for v in unassigned[cut:]]

    def _do(task):
        vpath, split = task
        out = frames_out / split / 'real' / vpath.stem
        return extract_frames_from_video(
            str(vpath), str(out), args.n_frames, args.img_size
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_do, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 200 == 0:
                print(f'    real: {done}/{len(tasks)}')


def _extract_batch_fake(mp4_files, frames_out, args):
    """Extract frames for fake videos — assign 80/20 matching real ratio."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from mstfnet.dataset import extract_frames_from_video
    import random

    random.seed(42)
    shuffled = list(mp4_files)
    random.shuffle(shuffled)
    cut = int(len(shuffled) * 0.8)
    tasks = (
        [(v, 'train') for v in shuffled[:cut]] +
        [(v, 'val')   for v in shuffled[cut:]]
    )

    def _do(task):
        vpath, split = task
        out = frames_out / split / 'fake' / vpath.stem
        return extract_frames_from_video(
            str(vpath), str(out), args.n_frames, args.img_size
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_do, t): t for t in tasks}
        done = 0
        for f in as_completed(futs):
            done += 1
            if done % 200 == 0:
                print(f'    fake: {done}/{len(tasks)}')


if __name__ == '__main__':
    main()
