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
import shutil

class InceptionV3FeatureExtractor(nn.Module):
    """Inception v3 model for extracting features at the pool3 layer"""
    def __init__(self):
        super().__init__()
        # Load pretrained Inception v3
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
        # Get features
        x = self.blocks(x)
        # Flatten
        x = torch.flatten(x, 1)
        return x

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
    
    # Clean up zip file to save space
    os.remove(zip_path)
    
    return val2014_dir

def get_coco_image_paths_from_csv(csv_path, coco_dir, max_images=None):
    """Get the actual COCO image paths based on the CSV file"""
    
    df = pd.read_csv(csv_path)
    
    # Use only first max_images if specified
    if max_images:
        df = df.head(max_images)
        print(f"Using first {max_images} images from CSV")
    
    image_ids = df['image_id'].tolist()
    
    image_paths = []
    missing = []
    
    for img_id in image_ids:
        # COCO images are named as COCO_val2014_000000XXXXXX.jpg
        filename = f"COCO_val2014_{img_id:012d}.jpg"
        filepath = os.path.join(coco_dir, filename)
        
        if os.path.exists(filepath):
            image_paths.append(filepath)
        else:
            missing.append(filename)
    
    print(f"Found {len(image_paths)} real COCO images out of {len(image_ids)}")
    if missing:
        print(f"Missing {len(missing)} images")
        if len(missing) <= 10:
            print(f"Missing files: {missing}")
    
    return image_paths

def get_generated_images_matching_csv(generated_dir, csv_path, max_images=None):
    """Get generated images that match the CSV image IDs"""
    
    df = pd.read_csv(csv_path)
    
    # Use only first max_images if specified
    if max_images:
        df = df.head(max_images)
        print(f"Using first {max_images} images from CSV for generated images")
    
    image_ids = df['image_id'].tolist()
    
    image_paths = []
    missing = []
    
    for img_id in image_ids:
        # Generated images are named as 000000000133.png
        filename = f"{img_id:012d}.png"
        filepath = os.path.join(generated_dir, filename)
        
        if os.path.exists(filepath):
            image_paths.append(filepath)
        else:
            missing.append(filename)
    
    print(f"Found {len(image_paths)} generated images out of {len(image_ids)} expected")
    if missing:
        print(f"Missing {len(missing)} generated images")
        if len(missing) <= 10:
            print(f"Missing files: {missing[:10]}")
    
    return image_paths

def calculate_activation_statistics_from_paths(image_paths, model, batch_size=32, device='cuda', max_images=None):
    """Calculate mean and covariance of features for specified image paths"""
    
    model.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
                features = model(batch_tensor)
                features_list.append(features.cpu().numpy())
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    
    # Calculate statistics
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Inception Distance between two distributions"""
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, f"Mean shapes don't match: {mu1.shape} vs {mu2.shape}"
    assert sigma1.shape == sigma2.shape, f"Covariance shapes don't match: {sigma1.shape} vs {sigma2.shape}"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return fid

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
    parser.add_argument("--use_30k_subset", action="store_true", help="Match images to CSV entries")
    
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
    
    # Handle real image statistics
    if args.real_mu and args.real_sigma and os.path.exists(args.real_mu) and os.path.exists(args.real_sigma):
        # Use precomputed statistics
        print(f"\nLoading precomputed real image statistics from {args.real_mu} and {args.real_sigma}")
        real_mu = np.load(args.real_mu)
        real_sigma = np.load(args.real_sigma)
        print(f"Real statistics shape - mu: {real_mu.shape}, sigma: {real_sigma.shape}")
    else:
        # Compute from real COCO images
        print("\nComputing real image statistics from COCO val2014...")
        
        # Download COCO val2014 if requested
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
        
        if args.use_30k_subset:
            # Get the specific image paths from CSV
            print(f"\nExtracting real images matching CSV entries...")
            image_paths = get_coco_image_paths_from_csv(args.csv_path, val2014_dir, max_images=args.max_images)
        else:
            # Get all images from directory
            image_files = [f for f in os.listdir(val2014_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if args.max_images:
                image_files = image_files[:args.max_images]
            image_paths = [os.path.join(val2014_dir, f) for f in image_files]
            print(f"Found {len(image_paths)} images in {val2014_dir}")
        
        if len(image_paths) == 0:
            print("Error: No matching COCO images found!")
            return
        
        real_mu, real_sigma = calculate_activation_statistics_from_paths(
            image_paths, 
            model, 
            batch_size=args.batch_size,
            device=device
        )
        
        # Save real statistics if requested
        if args.save_real_stats:
            np.save("real_mu_10k.npy", real_mu)
            np.save("real_sigma_10k.npy", real_sigma)
            print(f"Saved real statistics to real_mu_10k.npy and real_sigma_10k.npy")
    
    # Calculate statistics for generated images
    print(f"\nProcessing generated images...")
    
    if args.use_30k_subset:
        # Get only the generated images that match CSV IDs
        print(f"Matching generated images to CSV entries...")
        gen_image_paths = get_generated_images_matching_csv(
            args.generated_dir, 
            args.csv_path, 
            max_images=args.max_images
        )
        
        if len(gen_image_paths) == 0:
            print("Error: No matching generated images found!")
            return
            
        gen_mu, gen_sigma = calculate_activation_statistics_from_paths(
            gen_image_paths,
            model,
            batch_size=args.batch_size,
            device=device
        )
    else:
        # Get all images from generated directory
        image_files = [f for f in os.listdir(args.generated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if args.max_images:
            image_files = image_files[:args.max_images]
        gen_image_paths = [os.path.join(args.generated_dir, f) for f in image_files]
        print(f"Found {len(gen_image_paths)} generated images")
        
        gen_mu, gen_sigma = calculate_activation_statistics_from_paths(
            gen_image_paths,
            model,
            batch_size=args.batch_size,
            device=device
        )
    
    # Save generated statistics if requested
    if args.save_gen_stats:
        np.save("generated_mu_10k.npy", gen_mu)
        np.save("generated_sigma_10k.npy", gen_sigma)
        print(f"Saved generated statistics to generated_mu_10k.npy and generated_sigma_10k.npy")
    
    # Calculate FID
    print("\nCalculating FID score...")
    fid_score = calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)
    
    print(f"\n{'='*60}")
    print(f"FID Score: {fid_score:.2f}")
    print(f"{'='*60}")
    
    # Additional info
    print(f"\nDetails:")
    print(f"  Real images: shape of mu={real_mu.shape}, sigma={real_sigma.shape}")
    print(f"  Generated images: shape of mu={gen_mu.shape}, sigma={gen_sigma.shape}")
    if args.max_images:
        print(f"  Number of samples used: {args.max_images}")
    
    return fid_score

if __name__ == "__main__":
    main()