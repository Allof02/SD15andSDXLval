# from cleanfid import fid



# score = fid.compute_fid(
#     gen, real,               
#     mode="clean",
#     model_name="inception_v3",
#     batch_size=64,
#     num_workers=0,    
#     device="cuda"
# )
# print("Clean-FID:", score)

# fid_pytorchfid_fixed.py
import argparse, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

real = r"D:\datasets\COCO\val2014_selected"
gen  = r"D:\VSC_research\generative-models\sdxl\generated_image_sdxl_512_sc1024_10k"

class ImageFolderList(Dataset):
    def __init__(self, root: Path, exts=(".png", ".jpg", ".jpeg", ".webp")):
        self.paths = [p for p in Path(root).iterdir() if p.suffix.lower() in exts]
        self.transform = transforms.Compose([
            transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(299),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def activations_for_dir(dir_path: str, device: str, batch_size: int) -> np.ndarray:
    """Returns Nx2048 Inception features for all images in a folder."""
    dataset = ImageFolderList(Path(dir_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()

    feats = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        # T.F normalize to [-1,1] like pytorch-fid expects
        x = x * 2 - 1
        pred = model(x)[0]  # [B, 2048, 1, 1]
        feats.append(pred.squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--real_dir", required=True)
    # ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    real_acts = activations_for_dir(real, args.device, args.batch_size)
    gen_acts  = activations_for_dir(gen,  args.device, args.batch_size)

    m1, s1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    m2, s2 = gen_acts.mean(axis=0),  np.cov(gen_acts,  rowvar=False)
    fid = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()

