#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import List, Dict
import concurrent.futures as futures

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

try:
    from scipy import linalg
except Exception as e:
    raise SystemExit("Please install SciPy: pip install scipy") from e

try:
    import requests
except Exception as e:
    raise SystemExit("Please install requests: pip install requests") from e

ID_RE = re.compile(r"(\d{12})\.png$", re.IGNORECASE)

def find_generated(gen_dir: Path) -> Dict[int, Path]:
    out = {}
    for p in gen_dir.glob("*.png"):
        # if len(out) == 1000:
        #     break
        m = ID_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out

def expected_coco_path(real_dir: Path, img_id: int) -> Path:
    return real_dir / f"COCO_val2014_{img_id:012d}.jpg"

def coco_val_url(img_id: int, base: str) -> str:
    # Official pattern: http(s)://images.cocodataset.org/val2014/COCO_val2014_<12d>.jpg
    return f"{base.rstrip('/')}/COCO_val2014_{img_id:012d}.jpg"

def download_one(img_id: int, dst: Path, base_url: str, timeout: float = 30.0) -> bool:
    url = coco_val_url(img_id, base_url)
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        if r.status_code == 200:
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 14):
                    if chunk:
                        f.write(chunk)
            return True
        return False
    except Exception:
        return False

def ensure_reals(ids: List[int], real_dir: Path, base_url: str, max_workers: int = 8):
    missing = [i for i in ids if not expected_coco_path(real_dir, i).exists()]
    if not missing:
        return
    print(f"Downloading {len(missing)} missing COCO val2014 images to {real_dir} ...")
    with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        tasks = []
        for i in missing:
            dst = expected_coco_path(real_dir, i)
            tasks.append(ex.submit(download_one, i, dst, base_url))
        ok = 0
        for done in tqdm(futures.as_completed(tasks), total=len(tasks), desc="Downloading"):
            ok += 1 if done.result() else 0
    have = sum(expected_coco_path(real_dir, i).exists() for i in ids)
    print(f"Done. Available real images: {have}/{len(ids)}")

class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

@torch.no_grad()
def compute_acts(paths: List[Path], model: nn.Module, transform, bs: int, nw: int, device: str) -> np.ndarray:
    dl = DataLoader(ImagePathDataset(paths, transform),
                    batch_size=bs, shuffle=False, num_workers=nw,
                    pin_memory=True, drop_last=False)
    feats = []
    for x in tqdm(dl, desc="Extracting features", leave=False):
        x = x.to(device, non_blocking=True)
        f = model(x)  # [B, 2048]
        if hasattr(f, "logits"):   # torchvision InceptionOutputs
            f = f.logits
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def fid(mu1, sig1, mu2, sig2) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig1 + sig2 - 2.0 * covmean))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="Dir of generated PNGs (000000xxxxxx.png)")
    ap.add_argument("--real_dir", required=True, help="Dir to find/store matched COCO reals (JPGs)")
    ap.add_argument("--download_missing", action="store_true", help="Download any missing real images")
    ap.add_argument("--coco_base_url", default="http://images.cocodataset.org/val2014",
                    help="Base URL hosting COCO val2014 images")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    real_dir = Path(args.real_dir)
    if not gen_dir.exists():
        raise SystemExit(f"Missing gen_dir: {gen_dir}")
    real_dir.mkdir(parents=True, exist_ok=True)

    gen_map = find_generated(gen_dir)
    if not gen_map:
        raise SystemExit("No generated PNGs found (expected 12-digit ids).")

    gen_ids = sorted(gen_map.keys())
    print(f"Generated images: {len(gen_ids)}")

    if args.download_missing:
        ensure_reals(gen_ids, real_dir, args.coco_base_url)

    # Intersect by presence on disk
    common_ids = [i for i in gen_ids if expected_coco_path(real_dir, i).exists()]
    if not common_ids:
        raise SystemExit("No matching real images found on disk. Use --download_missing or point --real_dir to your COCO val2014.")
    print(f"Using intersection: {len(common_ids)} images")

    gen_paths  = [gen_map[i] for i in common_ids]
    real_paths = [expected_coco_path(real_dir, i) for i in common_ids]

    # Inception-V3 (ImageNet weights), use pooling head (2048-D)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(args.device)

    transform = weights.transforms()  # 299 crop + normalize

    print("\nReal features:")
    real_feats = compute_acts(real_paths, model, transform, args.batch_size, args.num_workers, args.device)
    print("Generated features:")
    gen_feats  = compute_acts(gen_paths,  model, transform, args.batch_size, args.num_workers, args.device)

    mu_r, mu_g = real_feats.mean(0), gen_feats.mean(0)
    sig_r, sig_g = np.cov(real_feats, rowvar=False), np.cov(gen_feats, rowvar=False)

    score = fid(mu_r, sig_r, mu_g, sig_g)
    print(f"\nFID on {len(common_ids)} matched images: {score:.4f}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    main()

# (sd-xl) D:\generative-models\sdxl>python .\compute_fid_coco.py --gen_dir D:\VSC_research\generative-models\sdxl\generated_image_sdxl_512_sc1024_10k --real_dir D:\datasets\COCO\val2014_selected --download_missing --batch_size 32 --num_workers 4
# Generated images: 3500
# Using intersection: 3500 images
# Downloading: "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth" to C:\Users\petec/.cache\torch\hub\checkpoints\inception_v3_google-0cc3c7bd.pth
# 100%|████████████████████████████████████████████████████████████████████████████████| 104M/104M [00:00<00:00, 119MB/s]

# 3500 Generated images vs. 3500 Real images
# FID on 3500 matched images: 103.2066



