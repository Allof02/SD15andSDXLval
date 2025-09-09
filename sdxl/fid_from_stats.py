#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights

try:
    from scipy import linalg
except Exception as e:
    raise SystemExit("Please install SciPy: pip install scipy") from e


def list_images(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


class ImageFolderDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def compute_activations(paths: List[Path], model: nn.Module, transform, batch_size: int,
                        num_workers: int, device: str) -> np.ndarray:
    dl = DataLoader(
        ImageFolderDataset(paths, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    feats = []
    for x in tqdm(dl, desc="Extracting features", leave=False):
        x = x.to(device, non_blocking=True)
        out = model(x)
        if hasattr(out, "logits"):
            out = out.logits
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def load_stats(npz_path: Path):
    data = np.load(npz_path)
    keys = set(data.keys())
    if {"mu", "sigma"} <= keys:
        mu, sigma = data["mu"], data["sigma"]
    elif {"mu_r", "sigma_r"} <= keys:
        mu, sigma = data["mu_r"], data["sigma_r"]
    elif {"feat_mu", "feat_sigma"} <= keys:
        mu, sigma = data["feat_mu"], data["feat_sigma"]
    else:
        raise SystemExit(f"Could not find (mu, sigma) in {npz_path}. Keys present: {sorted(keys)}")
    return mu, sigma, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, help="Directory of generated images")
    ap.add_argument("--stats_npz", required=True, help="Path to precomputed stats .npz (contains mu & sigma)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    stats_npz = Path(args.stats_npz)
    if not gen_dir.exists():
        raise SystemExit(f"gen_dir not found: {gen_dir}")
    if not stats_npz.exists():
        raise SystemExit(f"stats_npz not found: {stats_npz}")

    # Load real stats
    mu_r, sigma_r, keys = load_stats(stats_npz)
    print(f"Loaded stats from {stats_npz.name} with keys {sorted(keys)}")
    print(f"mu shape: {mu_r.shape}, sigma shape: {sigma_r.shape}")

    # Discover generated images
    gen_paths = list_images(gen_dir)
    if not gen_paths:
        raise SystemExit(f"No images found in {gen_dir}")
    print(f"Found {len(gen_paths)} generated images")

    # Build Inception-V3 with torchvision weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights)  # do NOT pass aux_logits=False
    model.fc = nn.Identity()               # use pooled 2048-D features
    model.eval().to(args.device)
    transform = weights.transforms()

    # Compute activations for generated images
    gen_feats = compute_activations(gen_paths, model, transform,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    device=args.device)

    # Sanity: dimensions should match stats (usually 2048)
    if gen_feats.shape[1] != mu_r.shape[0]:
        raise SystemExit(
            f"Feature dimension mismatch: gen_feats={gen_feats.shape[1]} vs stats={mu_r.shape[0]}.\n"
            "Your stats may have been computed with a different backbone/library."
        )

    # Compute gen stats and FID
    mu_g = gen_feats.mean(axis=0)
    sigma_g = np.cov(gen_feats, rowvar=False)
    fid = calculate_fid(mu_r, sigma_r, mu_g, sigma_g)
    print(f"\nFID (generated vs stats): {fid:.4f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    main()

# Loaded stats from MS-COCO_val2014_fid_stats.npz with keys ['mu', 'sigma']
# mu shape: (2048,), sigma shape: (2048, 2048)
# Found 3500 generated images
# 3500 generated images vs. Distribution of 30K real images
# FID (generated vs stats): 245.2440

# python .\fid_from_stats.py --gen_dir D:\VSC_research\generative-models\sdxl\generated_image_sdxl_512_no_sc_3_5k --stats_npz D:\VSC_research\generative-models\sdxl\MS-COCO_val2014_fid_stats.npz --batch_size 32 --num_workers 4 --device cuda