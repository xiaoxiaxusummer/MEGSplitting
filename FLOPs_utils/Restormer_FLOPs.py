import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np

from thop import profile
# from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt

import os, sys
PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__))+'/../')
sys.path.append(PROJECT_DIR)
SAVE_DIR = os.path.join(PROJECT_DIR, "samples/")
os.makedirs(SAVE_DIR, exist_ok=True)

model_name = 'Restormer'
if model_name == 'Restormer':
    task = 'Single_Image_Defocus_Deblurring'
    def get_weights_and_parameters(task, parameters):
        if task == 'Motion_Deblurring':
            weights = os.path.join(f'{PROJECT_DIR}/Restormer/Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
        elif task == 'Single_Image_Defocus_Deblurring':
            weights = os.path.join(f'{PROJECT_DIR}/Restormer/Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
        elif task == 'Deraining':
            weights = os.path.join(f'{PROJECT_DIR}/Restormer/Deraining', 'pretrained_models', 'deraining.pth')
        elif task == 'Real_Denoising':
            weights = os.path.join(f'{PROJECT_DIR}/Restormer/Denoising', 'pretrained_models', 'real_denoising.pth')
            parameters['LayerNorm_type'] =  'BiasFree'
        return weights, parameters
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    weights, parameters = get_weights_and_parameters(task, parameters)
    load_arch = run_path(os.path.join(f'{PROJECT_DIR}/Restormer/basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
    model.to('cuda', torch.float16).eval()


sizes = torch.arange(64, 1024+64, 64)
areas = sizes**2
FLOPs_all = []

with torch.no_grad():
    for sz in sizes:
        input_tensor = torch.randn(1, 3, sz, sz).to('cuda', torch.float16)
        flops, params = profile(model, inputs=(input_tensor,), verbose =False)
        # flops, params = get_model_complexity_info(model, (3, sz, sz), as_strings=False, verbose=False)
        FLOPs_all.append(flops/1e9)
        print(f"Input: {sz}x{sz} → FLOPs: {flops / 1e9:.2f} GFLOPs")

plt.figure(figsize=(6, 4))
plt.plot(areas, FLOPs_all, marker='o', label='Restormer')
plt.title("Restormer FLOPs vs Image Size")
plt.xlabel(r"Image size ($HW$)")
plt.ylabel("Model complexity (GFLOPs)")
plt.xticks(areas, [f"{s}x{s}" for s in sizes])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
