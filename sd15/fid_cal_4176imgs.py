import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
from scipy import linalg
import argparse
import urllib.request
import zipfile

# ------------------------------
# Feature extractor (torchvision Inception v3)
# ------------------------------
class InceptionV3FeatureExtractor(nn.Module):
    """Inception v3 model for extracting features at the pool3 layer"""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        inception.eval()

        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x):
        # Resize to 299x299 as required by Inception v3
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.blocks(x)        # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)   # (B, 2048)
        return x

# ------------------------------
# COCO download helper
# ------------------------------
def download_coco_val2014(target_dir="coco_val2014"):
    """Download COCO val2014 images if not already present"""
    val2014_url = "http://images.cocodataset.org/zips/val2014.zip"
    val2014_dir = os.path.join(target_dir, "val2014")

    if os.path.exists(val2014_dir) and len(os.listdir(val2014_dir)) > 40000:
        print(f"COCO val2014 images already exist in {val2014_dir}")
        return val2014_dir

    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "val2014.zip")

    # Download if not exists
    if not os.path.exists(zip_path):
        print(f"Downloading COCO val2014 images (6GB)...")
        print(f"From: {val2014_url}")

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"Download progress: {percent:.1f}%", end='\r')

        urllib.request.urlretrieve(val2014_url, zip_path, download_progress)
        print("\nDownload complete!")

    # Extract
    print("Extracting val2014.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction complete!")

    # Keep the zip or delete to save space; your original removed it:
    # os.remove(zip_path)

    return val2014_dir

# ------------------------------
# ID and path helpers (for exact matching)
# ------------------------------
def get_ids_from_csv(csv_path, max_images=None):
    """Read image_id column (ordered) from CSV."""
    df = pd.read_csv(csv_path)
    if max_images:
        df = df.head(max_images)
    return df['image_id'].tolist()

def list_existing_generated_ids(generated_dir):
    """Return a set of IDs inferred from files like 000000123456.png/.jpg/.jpeg"""
    ids = set()
    for f in os.listdir(generated_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            stem = os.path.splitext(f)[0]
            if len(stem) == 12 and stem.isdigit():
                ids.add(int(stem))
    return ids

def build_paths_for_ids(ids, coco_val2014_dir, generated_dir):
    """Build parallel real/gen paths for the SAME list of IDs."""
    real_paths = [os.path.join(coco_val2014_dir, f"COCO_val2014_{i:012d}.jpg") for i in ids]
    gen_paths  = [os.path.join(generated_dir, f"{i:012d}.png") for i in ids]
    return real_paths, gen_paths

# ------------------------------
# Generic path collectors (non-subset mode)
# ------------------------------
def get_coco_image_paths_from_dir(val2014_dir, max_images=None):
    image_files = [f for f in os.listdir(val2014_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    if max_images:
        image_files = image_files[:max_images]
    return [os.path.join(val2014_dir, f) for f in image_files]

def get_generated_images_from_dir(generated_dir, max_images=None):
    image_files = [f for f in os.listdir(generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    if max_images:
        image_files = image_files[:max_images]
    return [os.path.join(generated_dir, f) for f in image_files]

# ------------------------------
# Feature extraction & stats
# ------------------------------
def calculate_activation_statistics_from_paths(image_paths, model, batch_size=32, device='cuda', max_images=None):
    """Calculate mean and covariance of features for specified image paths"""
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if max_images:
        image_paths = image_paths[:max_images]

    print(f"Processing {len(image_paths)} images...")

    features_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(device)
                feats = model(batch_tensor)          # (B, 2048)
                features_list.append(feats.cpu().numpy())

    if not features_list:
        raise RuntimeError("No features extracted; check your image paths and formats.")

    all_features = np.concatenate(features_list, axis=0)
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    return mu, sigma

# ------------------------------
# FID
# ------------------------------
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Inception Distance between two distributions"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, f"Mean shapes don't match: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Covariance shapes don't match: {sigma1.shape} vs {sigma2.shape}"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calculate FID score with COCO val2014")
    parser.add_argument("--generated_dir", required=True, help="Directory containing generated images")
    parser.add_argument("--csv_path", default="MS-COCO_val2014_30k_captions.csv", help="Path to CSV with image IDs")
    parser.add_argument("--real_mu", default=None, help="Path to precomputed real images mean (optional)")
    parser.add_argument("--real_sigma", default=None, help="Path to precomputed real images covariance (optional)")
    parser.add_argument("--coco_dir", default="coco_val2014", help="Directory to store/find COCO val2014 images")
    parser.add_argument("--download", action="store_true", help="Download COCO val2014 if not present")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--save_real_stats", action="store_true", help="Save computed real image statistics")
    parser.add_argument("--save_gen_stats", action="store_true", help="Save generated image statistics")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--use_30k_subset", action="store_true", help="Match images to CSV entries (ID-aligned)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("GPU NOT AVAILABLE. USING CPU")

    # Initialize Inception v3 model
    print("Loading Inception v3 model...")
    model = InceptionV3FeatureExtractor().to(device)

    print(f"\nConfiguration:")
    print(f"  Generated images dir: {args.generated_dir}")
    print(f"  CSV path: {args.csv_path}")
    if args.max_images:
        print(f"  Using first {args.max_images} samples from CSV")

    # Resolve COCO val dir
    if args.real_mu and args.real_sigma and os.path.exists(args.real_mu) and os.path.exists(args.real_sigma):
        val2014_dir = os.path.join(args.coco_dir, "val2014")  # May be unused if we rely solely on precomputed
    else:
        if args.download:
            val2014_dir = download_coco_val2014(args.coco_dir)
        else:
            val2014_dir = os.path.join(args.coco_dir, "val2014")
            if not os.path.exists(val2014_dir):
                print(f"Error: COCO val2014 directory not found at {val2014_dir}")
                print("Use --download flag to download automatically, or download manually:")
                print("wget http://images.cocodataset.org/zips/val2014.zip")
                print("unzip val2014.zip -d coco_val2014/")
                return

    # --------------------------
    # Build real/gen path lists
    # --------------------------
    image_paths = None       # real
    gen_image_paths = None   # generated

    if args.use_30k_subset:
        # Build by exact ID intersection (preserves CSV order)
        csv_ids = get_ids_from_csv(args.csv_path, max_images=None)  # don't cap yet
        gen_ids_present = list_existing_generated_ids(args.generated_dir)

        matched_ids = [i for i in csv_ids if i in gen_ids_present]

        # Target count
        target_n = args.max_images if args.max_images else len(matched_ids)
        if len(matched_ids) < target_n:
            print(f"Warning: only {len(matched_ids)} generated images match CSV IDs; requested {target_n}. Using {len(matched_ids)}.")
            target_n = len(matched_ids)
        matched_ids = matched_ids[:target_n]

        image_paths, gen_image_paths = build_paths_for_ids(matched_ids, val2014_dir, args.generated_dir)

        # Safety: drop any missing files on disk
        image_paths = [p for p in image_paths if os.path.exists(p)]
        gen_image_paths = [p for p in gen_image_paths if os.path.exists(p)]

        if len(image_paths) == 0 or len(gen_image_paths) == 0:
            print("Error: No matching COCO or generated images found after file checks!")
            return

        if len(image_paths) != len(gen_image_paths):
            common = min(len(image_paths), len(gen_image_paths))
            print(f"After file checks: real={len(image_paths)}, gen={len(gen_image_paths)}; truncating both to {common}")
            image_paths = image_paths[:common]
            gen_image_paths = gen_image_paths[:common]

        print(f"Final matched pair count: {len(image_paths)}")

    else:
        # Non-subset mode: just list both dirs (sorted) and optionally cap
        image_paths = get_coco_image_paths_from_dir(val2014_dir, max_images=args.max_images)
        gen_image_paths = get_generated_images_from_dir(args.generated_dir, max_images=args.max_images)
        print(f"Found {len(image_paths)} real images and {len(gen_image_paths)} generated images (non-subset mode)")
        # Not enforcing same count here; FID doesn't require pairs. You may slice to min if desired.

    # --------------------------
    # Real stats (either precomputed or computed now)
    # --------------------------
    if args.real_mu and args.real_sigma and os.path.exists(args.real_mu) and os.path.exists(args.real_sigma):
        print(f"\nLoading precomputed real image statistics from {args.real_mu} and {args.real_sigma}")
        real_mu = np.load(args.real_mu)
        real_sigma = np.load(args.real_sigma)
        print(f"Real statistics shape - mu: {real_mu.shape}, sigma: {real_sigma.shape}")
        n_real_used = None  # unknown from files
    else:
        print("\nComputing real image statistics from COCO val2014...")
        real_mu, real_sigma = calculate_activation_statistics_from_paths(
            image_paths,
            model,
            batch_size=args.batch_size,
            device=device
        )
        n_real_used = len(image_paths)
        if args.save_real_stats:
            np.save(f"real_mu_{n_real_used}.npy", real_mu)
            np.save(f"real_sigma_{n_real_used}.npy", real_sigma)
            print(f"Saved real statistics to real_mu_{n_real_used}.npy and real_sigma_{n_real_used}.npy")

    # --------------------------
    # Generated stats
    # --------------------------
    print(f"\nProcessing generated images...")
    gen_mu, gen_sigma = calculate_activation_statistics_from_paths(
        gen_image_paths,
        model,
        batch_size=args.batch_size,
        device=device
    )
    n_gen_used = len(gen_image_paths)
    if args.save_gen_stats:
        np.save(f"generated_mu_{n_gen_used}_512.npy", gen_mu)
        np.save(f"generated_sigma_{n_gen_used}_512.npy", gen_sigma)
        print(f"Saved generated statistics to generated_mu_{n_gen_used}_512.npy and generated_sigma_{n_gen_used}_512.npy")

    # --------------------------
    # FID
    # --------------------------
    print("\nCalculating FID score...")
    fid_score = calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)

    print(f"\n{'='*60}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'='*60}")

    print(f"\nDetails:")
    print(f"  Real images: mu={real_mu.shape}, sigma={real_sigma.shape}")
    print(f"  Generated images: mu={gen_mu.shape}, sigma={gen_sigma.shape}")
    if n_real_used is not None:
        print(f"  Real images used: {n_real_used}")
    print(f"  Generated images used: {n_gen_used}")

    return fid_score

if __name__ == "__main__":
    main()
