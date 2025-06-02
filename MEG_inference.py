import torch

from PIL import Image
import numpy as np
import random
import os, sys
import math

project_dir = os.path.abspath(os.path.dirname(__file__))
print(project_dir)
sys.path.append(project_dir)

# GPU_INDEX = 1  # GPU device to use
# torch.cuda.set_device(GPU_INDEX)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF']= 'expandable_segments:True'

TEST_MODE = 0 # 0 or 
ES_MODEL = 'SDXL'  # @param 'SD3', 'SDXL'
HF_LOGIN_TOKEN = None # Insert your hugging face login token here if use SD3
UE_MODEL = 'sdxl-union' # @param 'sdxl-union', 'sd3-controlnet', 
UNET_MODEL = 'SDXL-Lightning-2step' # @param None, 'SDXL-Lightning-2step', 'SDXL-Lightning-4step'

if ES_MODEL == 'SD3':
     assert HF_LOGIN_TOKEN is not None, "SD3-medium requires a hugging face login token" 


if TEST_MODE == 0:
     from MEG_utils.MEG_generation_utils_mode_0 import *
else:
     from MEG_utils.MEG_generation_utils_mode_1 import *
SEED = 111 if ES_MODEL=='SD3' else 222
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
########################### 1. Load Pipeline ##########################
pipe_ES = load_ES_pipeline(ES_MODEL, device, HF_LOGIN_TOKEN)
pipe_mobile = load_mobile_pipeline(UE_MODEL, UNET_MODEL=UNET_MODEL, device=device)

############################ 2. Image configuration ##########################
image_width, image_height = 1024, 1024
# prompt = "A picture of a beautiful Ragdoll cat with blue eyes standing in the snowy street"
prompt = "A painting of a beautiful girl standing in the snowy street"
for x in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: # ratio of ES generation
    setup_seed(SEED)
    torch.cuda.empty_cache()

    focus_width, focus_height = ratio_splitting(image_width, image_height, x)
    #### [Note] SD-style algorithm only supports a resolution that is a multiple of 8 
    # focus_width, focus_height = ceil_multiple_of_16([focus_width, focus_height])
    focus_width, focus_height = nearest_multiple_of_8([focus_width, focus_height])

    ################## 3. Perform collaborative generation ################
    create_full_image = True if (x==0.4 or TEST_MODE==1) else False
    center_image, focus_coords = ES_generation(x, ES_MODEL, pipe_ES, prompt, focus_width, focus_height, image_width, image_height, seed=SEED, create_full_image=create_full_image)
    Mobile_Generation(x, UE_MODEL, pipe_mobile, prompt, image_width, image_height, focus_width, focus_height, focus_coords, UNET_MODEL=UNET_MODEL)