import os, sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_DIR)
SAVE_DIR = os.path.join(PROJECT_DIR, "samples_SDXL/")
os.makedirs(SAVE_DIR, exist_ok=True)

from PIL import Image
import torch, torchvision
_ = torch.manual_seed(123)

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=64)

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def Calculate_SSIM(real_path,generated_path):
    """
    Args:
        img_dist_true: perfect img tensors in int8
        imgs_dist: generated img tensors in int8
    Returns:
        FID score
    """
    imgs_dist_true = torchvision.io.read_image(real_path).float() / 255.0  # 标准化到 [0,1]
    imgs_dist = torchvision.io.read_image(generated_path).float() / 255.0  # 标准化到 [0,1]
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_value = ssim_fn(imgs_dist_true.unsqueeze(0), imgs_dist.unsqueeze(0))
    return ssim_value.item()

def Calculate_FID(real_path,generated_path):
    """
    Args:
        img_dist_true: perfect img tensors in int8
        imgs_dist: generated img tensors in int8
    Returns:
        FID score
    """
    imgs_dist_true = torchvision.io.read_image(real_path)
    imgs_dist = torchvision.io.read_image(generated_path)
    fid.update(imgs_dist_true.expand(2,-1,-1,-1), real=True)
    fid.update(imgs_dist.expand(2,-1,-1,-1), real=False)
    fid_score = fid.compute()
    fid.reset()

    return fid_score

def Load_Imgs(img_files):
    imgs = []
    for p in img_files:
        imgs.append(torchvision.io.read_image(p))
    return torch.stack(imgs, 0)



for i in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    real_path = SAVE_DIR + f'full_image.png'
    generated_path = SAVE_DIR + f'final_image_{i}.png'
    fid_score = Calculate_FID(real_path, generated_path)
    ssim = Calculate_SSIM(real_path,generated_path)
    print(f'fid:{fid_score}, ssim{ssim}')