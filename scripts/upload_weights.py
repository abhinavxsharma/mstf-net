"""
scripts/upload_weights.py
=========================
Upload trained MSTF-Net weights to HuggingFace Hub.

Usage:
    export HF_TOKEN=your_token_here
    python scripts/upload_weights.py \
        --checkpoint_dir /content/drive/MyDrive/MSTF_checkpoints \
        --repo_id ixabhinavsharma/mstf-net

Authors : Abhinav Vats et al., Chandigarh University
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description='Upload weights to HuggingFace')
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--repo_id', default='ixabhinavsharma/mstf-net')
    p.add_argument('--results_dir', default=None)
    return p.parse_args()


def main():
    args  = parse_args()
    token = os.environ.get('HF_TOKEN')

    if not token:
        print('❌ HF_TOKEN not set. Run: export HF_TOKEN=your_token')
        return

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print('❌ huggingface_hub not installed. Run: pip install huggingface_hub')
        return

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(args.repo_id, token=token, exist_ok=True)
        print(f'✅ Repo ready: https://huggingface.co/{args.repo_id}')
    except Exception as e:
        print(f'Repo creation: {e}')

    # Upload all BEST checkpoints
    ckpt_dir   = Path(args.checkpoint_dir)
    best_ckpts = sorted(ckpt_dir.glob('*_BEST.pth'))

    if not best_ckpts:
        print(f'No *_BEST.pth found in {ckpt_dir}')
        # Upload all .pth files
        best_ckpts = sorted(ckpt_dir.glob('*.pth'))

    print(f'\nUploading {len(best_ckpts)} checkpoints...')

    for ckpt_path in best_ckpts:
        remote_path = f'weights/{ckpt_path.name}'
        print(f'  Uploading {ckpt_path.name} → {remote_path}')
        try:
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=remote_path,
                repo_id=args.repo_id,
                token=token,
            )
            print(f'  ✅ Done')
        except Exception as e:
            print(f'  ❌ Failed: {e}')

    # Upload results JSON if available
    if args.results_dir:
        results_dir = Path(args.results_dir)
        for jf in results_dir.glob('*.json'):
            print(f'  Uploading {jf.name}...')
            api.upload_file(
                path_or_fileobj=str(jf),
                path_in_repo=f'results/{jf.name}',
                repo_id=args.repo_id,
                token=token,
            )

    print(f'\n✅ All uploads complete')
    print(f'🔗 https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
