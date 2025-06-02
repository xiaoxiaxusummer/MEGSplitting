from PIL import Image
import numpy as np
import torch
import math
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from diffusers.models.model_loading_utils import load_state_dict
from diffusers import AutoencoderKL, LCMScheduler
from diffusers.utils import numpy_to_pil
from torchvision import transforms
from torchvision.transforms import ToPILImage
from basicsr.utils import img2tensor as _img2tensor
import cv2
import math, random
from runpy import run_path

import os, sys
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../'))
sys.path.append(PROJECT_DIR)
from MEG_utils.controlnet_union import ControlNetModel_Union
from MEG_utils.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline


SAVE_DIR = os.path.join(PROJECT_DIR, "samples/")
os.makedirs(SAVE_DIR, exist_ok=True)


def decode_latents_to_images(vae, latents):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return numpy_to_pil(image)

def nearest_multiple_of_8(x):
    return (8 * np.round(np.array(x) / 8)).astype(int).tolist()

def ceil_multiple_of_8(x):
    return (8 * np.ceil(np.array(x) / 8)).astype(int).tolist()


def nearest_multiple_of_16(x):
    return (16 * np.round(np.array(x) / 16)).astype(int).tolist()


def floor_multiple_of_16(x):
    return (8 * np.floor(np.array(x) / 8)).astype(int).tolist()


def ceil_multiple_of_16(x):
    return (16 * np.ceil(np.array(x) / 16)).astype(int).tolist()


def cal_focus_coords(width, height, focus_width, focus_height, ratio_left, ratio_top):
    crop_width = width - focus_width
    crop_height = height - focus_height
    left = nearest_multiple_of_8(crop_width * ratio_left)
    right = left + focus_width 
    top = nearest_multiple_of_8(crop_height * ratio_top)
    bottom = top + focus_height
    crop_coords = (left, top, right, bottom)
    return crop_coords

def crop_center_rectangle(image, focus_coords):
    """
    Crop out the central rectangular area of the image
    
    @ params
    - image: PIL Image
    - focus_width, focus_height
    - ratio_left, ratio_top: ratio of left/top parts in the remaining regions
    
    @ returns
    - the cropped focus image in the center (a PIL Image)
    """
    width, height = image.size
    return image.crop(focus_coords)

def create_image_with_center_rectangle(image_width, image_height, focus_image, focus_coords):
    # focus_width, focus_heitght = focus_image.size
    # Create a blank white image
    base_image = Image.new('RGB', (image_width, image_height), color='white')
    # Calculate the position of the rectangle (centered)
    left_top_x = focus_coords[0] # x -> left pixel (in width)
    left_top_y = focus_coords[1] # y -> upper pixel (in height) 
    base_image.paste(focus_image, (left_top_x, left_top_y))
    return base_image

def create_latent_with_center_rectangle(latent_width, latent_height, focus_coords, focus_latent):
    dtype, device = focus_latent.dtype, focus_latent.device 
    C = focus_latent.shape[1]
    partial_latent = torch.zeros((1, C, latent_width, latent_height), dtype=dtype, device=device)
    left_top_x, left_top_y = focus_coords[0], focus_coords[1]
    focus_latent_width, focus_latent_height = focus_latent.shape[-1], focus_latent.shape[-2]
    partial_latent[:,:,left_top_y:left_top_y+focus_latent_height,left_top_x:left_top_x+focus_latent_width] = focus_latent
    return partial_latent


def fill_center_rectangle_with_image(base_image, focus_image, focus_coords):
    """
    Fills the center rectangle of the base image with the fill image.

    Parameters:
        base_image (PIL Image): The original base image with a central rectangular area.
        focus_image (PIL Image): The image to fill the central rectangular area.
        focus_coords (4-tuple): define the coords of the focus image, (left, top, right bottom)

    Returns:
        PIL Image: The base image with the center rectangle filled.
    """
    # base_width, base_height = base_image.size
    left_top_x, left_top_y = focus_coords[0], focus_coords[1]
    base_image.paste(focus_image, (left_top_x, left_top_y))
    return base_image


def create_filtered_mask(focus_width, focus_height, ratio_top, margin_size=8):    
    gradient = np.ones((focus_width, focus_height), dtype=np.uint8)*255

    # left margin
    gradient[0:margin_size, :] = np.tile(np.linspace(50, 255, margin_size)[
                                            :, np.newaxis], (1, focus_height))
    # right margin
    gradient[-margin_size:, :] = np.tile(np.linspace(255, 50, margin_size)[
                                         :, np.newaxis], (1, focus_height))
    # top margin
    if ratio_top > 0:
        gradient[margin_size:, 0:margin_size] = np.tile(np.linspace(
            50, 255, margin_size)[np.newaxis, :], (focus_width-margin_size, 1))
    # bottom margin
    if ratio_top < 1:
        gradient[:, -margin_size:] = np.tile(np.linspace(
            255, 50, margin_size)[np.newaxis, :], (focus_width, 1))
    mask = Image.fromarray(gradient)
    # mask = mask.filter(ImageFilter.GaussianBlur(10))
    return mask

def ratio_splitting(image_width, image_height, x, aspect_ratio=1):
    """
    aspect_ratio: height/width
    """
    focus_area = image_width * image_height * x
    focus_height = math.sqrt(focus_area * aspect_ratio)
    focus_width = focus_height/aspect_ratio
    return focus_width, focus_height
    
def ES_create_image(ES_MODEL, pipe, prompt, seed, g_width, g_height):
    if ES_MODEL == 'SD3':
        num_inference_steps = 25
        with torch.no_grad():
            g_image = pipe(prompt=prompt,
                    num_inference_steps=num_inference_steps, width=g_width, height=g_height,
                    generator=torch.manual_seed(seed),
                    guidance_scale=7.0).images[0]
    elif ES_MODEL == 'SDXL':
        num_inference_steps = 25  # suggest steps >= num_win=8
        cfg_scale = 2.5  # suggest values [1.5, 2.0, 2.5]
        with torch.no_grad():
            g_image = pipe(
            prompt          = [f"photorealistic, uhd, high resolution, high quality, highly detailed; masterpiece, {prompt}"], 
            negative_prompt     = ["distorted, blur, low-quality, haze, out of focus"],
            height          = g_height,
            width           = g_width,
            num_inference_steps = num_inference_steps, 
            guidance_scale      = cfg_scale,
            generator=torch.manual_seed(seed),
            output_type     = 'pt',
            ).images[0]
            to_pil_image = ToPILImage()
            g_image = to_pil_image(g_image)
    elif ES_MODEL == 'PixArt-alpha':
        with torch.no_grad():
            g_image = pipe(
            prompt,
            height=g_height,
            width=g_width,
            generator=torch.manual_seed(seed),
            guidance_scale=0.5,  # 0.,
            num_inference_steps=25,
            ).images[0]
    else:
        raise NotImplementedError(f"Model {ES_MODEL} is not implemented")
    return g_image

def ES_generation(x, ES_MODEL, pipe, prompt, focus_width, focus_height, image_width, image_height, seed=111, create_full_image=False):

    if x > 0.5:
        ratio_left, ratio_top = 0.5, 1
    else:
        ratio_left, ratio_top = 0.5, 0.3
    focus_coords = cal_focus_coords(image_width, image_height, focus_width, focus_height, ratio_left, ratio_top)

    """ Create the full image as reference """
    if create_full_image:
        full_image = ES_create_image(ES_MODEL, pipe, prompt, seed, image_width, image_height)
        full_image.save(os.path.join(SAVE_DIR, "full_image.png")) 
    full_image = Image.open(os.path.join(SAVE_DIR, "full_image.png")).convert("RGB")

    """ Center image generation """
    focus_image = ES_create_image(ES_MODEL, pipe, prompt, seed, focus_width, focus_height)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda").eval()
    # vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae",torch_dtype=torch.float32).to("cuda").eval() 
    vae = vae.float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),                # → [0, 1]
        transforms.Normalize([0.5], [0.5])    # → [-1, 1]
    ])
    focus_tensor = preprocess(focus_image).unsqueeze(0).to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        focus_latent = vae.encode(focus_tensor).latent_dist.sample() * vae.config.scaling_factor
        decoded_focus = decode_latents_to_images(vae, focus_latent)[0]
        decoded_focus.save('test1.png')
        latent_width, latent_height = image_width // 8, image_height // 8
        focus_latent_coords = np.array(focus_coords) // 8
        filled_focus = create_latent_with_center_rectangle(latent_width, latent_height, focus_latent_coords, focus_latent)
        reconstructed_focus = decode_latents_to_images(vae, filled_focus)[0]
        reconstructed_focus = crop_center_rectangle(reconstructed_focus, focus_coords)
        reconstructed_focus.save('test.png')
    partial_image = create_image_with_center_rectangle(image_width, image_height, reconstructed_focus, focus_coords)
    reconstructed_focus.save(os.path.join(SAVE_DIR, "focus_image.png"))  # save
    partial_image.save(os.path.join(SAVE_DIR, f"partial_generated_image_{x}.png"))
    return focus_image, focus_coords


def Mobile_Generation(splitting_ratio, UE_MODEL, pipe, prompt, W, H, focus_width, focus_height, focus_coords, UNET_MODEL=None, center_image=None):
    ds_width, ds_height = 512, 512
    ds_ratio = ds_width/W
    ratio_left, ratio_top = focus_coords[0]/focus_width, focus_coords[1]/focus_height
    focus_latent_coords = np.array(focus_coords)//8
    focus_coarse_coords =(np.array(focus_coords) * ds_ratio).astype(int)
    # downsample to a smaller region to generate coarse prediction
    center_image = Image.open(os.path.join(SAVE_DIR, "focus_image.png")).convert("RGB")
    image = fill_center_rectangle_with_image(Image.fromarray(np.ones(
            (W, H, 3), dtype=np.uint8)*255), center_image, focus_coords)
    image_downsample = image.resize((ds_width, ds_height))
    if UE_MODEL == 'sdxl-union':
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(prompt, "cuda", True)
        cnet_image_downsample = image_downsample
        masked_coarse_coords = (np.array(focus_coords) * ds_ratio).astype(int)
        masked_coarse_coords[0:2] = masked_coarse_coords[0:2] + 4
        mask_downsample = fill_center_rectangle_with_image(Image.new('L', (ds_width, ds_height), 255),
                    Image.new('L', (int(focus_width*ds_ratio)-8, int(focus_height*ds_ratio)-8), 0),
                    masked_coarse_coords)
        cnet_image_downsample.paste(0, (0, 0), mask_downsample)
        guidance_scale = 1.4 if splitting_ratio > 0.5 else 1.5
        seed = 1
        with torch.no_grad():
            if UNET_MODEL=='SDXL-Lightning-2step':
                res_image = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    image=cnet_image_downsample,
                    num_inference_steps=2, guidance_scale=guidance_scale,
                    generator=torch.manual_seed(seed),
                )
            elif UNET_MODEL=='SDXL-Lightning-1step':
                res_image = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    image=cnet_image_downsample,
                    num_inference_steps=1, guidance_scale=1.16, 
                )
    res_image = res_image.resize((W, H))
    # res_image.save(f'{SAVE_DIR}/coarse_image1.png')
    res_image2 = match_histogram_color(np.array(res_image.convert('RGB')), np.array(center_image), 2)
    res_image2 = Image.fromarray(res_image2)
    task = 'Single_Image_Defocus_Deblurring'
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    weights, parameters = get_weights_and_parameters(task, parameters)

    load_arch = run_path(os.path.join(f'{PROJECT_DIR}/Restormer/basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
    model.cuda()

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['params'])
    model.eval()

    print(f"\n ==> Running {task} with weights {weights}\n ")
    margin_size = 32

    blended = Image.blend(res_image, res_image2.convert('RGBA'), alpha=0.3)
    blend_image = blended.copy()
    filtered_mask = fill_center_rectangle_with_image(Image.new('L', (W, H), 0),
                    create_filtered_mask(focus_width, focus_height, ratio_top, margin_size=min(16, focus_width, focus_height)),
                    focus_coords)
    blend_image.paste(image, (0, 0), filtered_mask)
    coarse = torch.from_numpy(np.array(blend_image.convert('RGB'))).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()
    focus_image = cv2.imread(os.path.join(SAVE_DIR, "focus_image.png"))
    focus_image = torch.from_numpy(cv2.cvtColor(focus_image, cv2.COLOR_BGR2RGB)).float().div(255.).permute(2,0,1).unsqueeze(0).cuda()
    

    left, top, right, bottom = focus_coords
    tile_left = coarse[:,:,0:H, 0:left+margin_size]
    tile_right = coarse[:,:,0:H, right-margin_size:]
    if ratio_top > 0:
        tile_top = coarse[:,:,0:min(top+margin_size,H), left:right]
    else:
        tile_top = None
    if ratio_top < 1: 
        tile_bottom = coarse[:,:,min(bottom-margin_size,H):H, left:right]
    else:
        tile_bottom = None
    tiles = [tile_left, tile_right, tile_top, tile_bottom]
    restoreds = []
    with torch.no_grad():
        for i, tile in enumerate(tiles):
            restored1 = perform_Restormer(model, tile)
            restored = restored1.permute(0, 2, 3, 1).cpu().detach().numpy()*255
            restoreds.append(restored[0])
    output = np.array(blend_image.convert('RGB'))
    output[0:H, 0:left+margin_size] = restoreds[0]
    output[0:H, right-margin_size:] = restoreds[1]
    if restoreds[2] is not None:
        output[0:min(top+margin_size,H), left:right] = restoreds[2]
    if restoreds[3] is not None:
        output[bottom-margin_size:H, left:right] = restoreds[3]
    full_image = Image.fromarray(output, 'RGB')
    filtered_mask = fill_center_rectangle_with_image(Image.new('L', (W, H), 0),
                    create_filtered_mask(focus_width, focus_height, ratio_top, margin_size=32),
                    focus_coords)
    full_image.paste(blend_image, (0, 0), filtered_mask)
    full_image.save(os.path.join(SAVE_DIR, f'final_image_{splitting_ratio}.png'))


def load_ES_pipeline(ES_MODEL, device='cuda',  HF_login_token = None):
    """ ------------------------- Load ES model ---------------------- """
    if ES_MODEL == 'SD3':
        from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline
        from huggingface_hub import login
        login(token=HF_login_token)
        pipe_ES = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
        # pipe_ES = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float16)
        pipe_ES.load_lora_weights('erikhsos/campusbier-sd3-lora-prompt', weight_name='pytorch_lora_weights.safetensors')
        pipe_ES.to(device)
    elif ES_MODEL == 'SDXL':
        from diffusers import StableDiffusionXLPipeline
        # from MEGUtils.pipeline_SDXL_FLOPs import StableDiffusionXLPipeline
        pipe_ES = StableDiffusionXLPipeline.from_pretrained("hansyan/perflow-sdxl-dreamshaper", torch_dtype=torch.float16, use_safetensors=True, variant="v0-fix")
        from MEG_utils.scheduler_perflow import PeRFlowScheduler
        pipe_ES.scheduler = PeRFlowScheduler.from_config(pipe_ES.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
        pipe_ES.to("cuda", torch.float16)
    elif ES_MODEL == 'PixArt-alpha':
        from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL, Transformer2DModel
        pipe_ES = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True)
        # use DALL-E 3 Consistency Decoder
        pipe_ES.vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder", torch_dtype=torch.float16)
        # use SA-Solver sampler
        from diffusers.schedulers.scheduling_sasolver import SASolverScheduler
        pipe_ES.scheduler = SASolverScheduler.from_config(
            pipe_ES.scheduler.config, algorithm_type='data_prediction')
        pipe_ES.text_encoder.to_bettertransformer()  # speed-up T5
        pipe_ES.to(device)
    else:
        raise NotImplementedError(f"Model {ES_MODEL} is not implemented")
    return pipe_ES


def load_mobile_pipeline(UE_MODEL='union', UNET_MODEL = None, device='cuda'):
    """-------------------------  Load UE model ---------------------"""
    if UE_MODEL == "sdxl-union":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(device)
        ######################### ControlNet: Union #########################
        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_union = ControlNetModel_Union.from_config(config)
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        controlnet, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_union, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
        )
        controlnet.to(device=device, dtype=torch.float16)

        ################# If utilize RealVisXL_Lightning as UNet ##############
        # from diffusers import TCDScheduler
        # pipe_MU = StableDiffusionXLFillPipeline.from_pretrained(
        #     "SG161222/RealVisXL_V5.0_Lightning",
        #     torch_dtype=torch.float16,
        #     vae=vae,
        #     controlnet=controlnet,
        #     variant="fp16",
        # ).to("cuda")
        # pipe_MU.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        ################# If utilize SDXL-Lightning as UNet ##################
        if UNET_MODEL=='SDXL-Lightning-1step':
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            # Use the correct ckpt for your step setting!
            # ckpt = "sdxl_lightning_2step_unet.safetensors"
            ckpt = "sdxl_lightning_1step_unet_x0.safetensors"
            unet = UNet2DConditionModel.from_config(
                base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(
                load_file(hf_hub_download(repo, ckpt), device="cuda"))
            pipe_MU = StableDiffusionXLFillPipeline.from_pretrained(
                base, unet=unet, vae=vae,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16").to("cuda")
            pipe_MU.scheduler = EulerDiscreteScheduler.from_config(
                pipe_MU.scheduler.config, timestep_spacing="trailing", prediction_type="sample")
        elif UNET_MODEL=='SDXL-Lightning-2step':
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            # Use the correct ckpt for your step setting!
            ckpt = "sdxl_lightning_2step_unet.safetensors"
            unet = UNet2DConditionModel.from_config(
                base, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
            pipe_MU = StableDiffusionXLFillPipeline.from_pretrained(
                base, unet=unet, vae=vae,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16").to("cuda")
            pipe_MU.scheduler = EulerDiscreteScheduler.from_config(
                pipe_MU.scheduler.config, timestep_spacing="trailing")
        return pipe_MU



def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

from skimage.exposure import match_histograms
def match_histogram_color(source, target, channel_axis=0):
    source = source.astype(np.float32)
    target = target.astype(np.float32)
    
    matched = match_histograms(source, target, channel_axis=channel_axis)
    return np.clip(matched, 0, 255).astype(np.uint8)


def get_weights_and_parameters(task, parameters):
    # if task == 'Motion_Deblurring':
    #     weights = os.path.join(f'{PROJECT_DIR}/Restormer/Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    if task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join(f'{PROJECT_DIR}/Restormer/Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join(f'{PROJECT_DIR}/Restormer/Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join(f'{PROJECT_DIR}/Restormer/Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

def perform_Restormer(model, input_):
    img_multiple_of = 8
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    # Pad the input if not_multiple_of 8
    h,w = input_.shape[2], input_.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    input_ = torch.nn.functional.pad(input_, (0,padw,0,padh), 'reflect')

    restored = model(input_)
    restored = torch.clamp(restored, 0, 1)

    # Unpad the output
    restored = restored[:,:,:h,:w]
    return restored

