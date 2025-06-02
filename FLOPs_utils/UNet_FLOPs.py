import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from thop import profile


def remove_thop_attributes(model):
    for module in model.modules():
        if hasattr(module, 'total_ops'):
            del module.total_ops
        if hasattr(module, 'total_params'):
            del module.total_params

###################### 1. FLOPs of VAE Decoder ===============
def flops_vae():
    sample = torch.randn(1, 4, 128, 128).to("cuda", torch.float16)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to('cuda').eval()
    # wrapper to only run decode
    class VAEDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae._decode(latents).sample
    vae_decoder = VAEDecoder(vae)
    flops, params = profile(vae_decoder, inputs=(sample,), verbose=False)
    print(f"VAE Decoder FLOPs: {flops/1e12:.4f} TFLOPs")
    print(f"Params: {params/1e6:.4f} MParams")
    remove_thop_attributes(vae_decoder)
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from olvae.utils import instantiate_from_config
    olitevae = load_model_from_config(config_path="configs/olitevaeB_im_f8c12.yaml", 
                                    ckpt_path="olitevaeB_im_f8c12.safetensors")



def flops_text():
    # ========================2. FLOPs of Text Encoder ===============
    # Load text_encoder_1 
    text_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder").to('cuda').eval()
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    text_model2 = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2").to('cuda').eval()
    tokenizer2 = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")
    # token ids
    inputs1 = tokenizer("A high quality photo of a beautiful cat wearing a red skirt and singing a song", return_tensors="pt", padding="max_length", max_length=77)
    input_ids1 = inputs1.input_ids.to('cuda')
    inputs2 = tokenizer2("A high quality photo of a beautiful cat wearing a red skirt and singing a song", return_tensors="pt", padding="max_length", max_length=77)
    input_ids2 = inputs2.input_ids.to('cuda')
    # Test FLOPs
    flops, params = profile(text_model, inputs=(input_ids1,), verbose=False)
    print(f"Text Encoder 1 FLOPs: {flops/1e12:.4f} TFLOPs")
    print(f"Params: {params/1e6:.4f} MParams")
    flops, params = profile(text_model2, inputs=(input_ids2,), verbose=False)
    print(f"Text Encoder 2 FLOPs: {flops/1e12:.4f} TFLOPs")
    print(f"Params: {params/1e6:.4f} MParams")
    remove_thop_attributes(text_model)
    remove_thop_attributes(text_model2)


def flops_unet_singlestep():
    ###########################3. FLOPs of UNet ===================
    model = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet").to("cuda")
    model.eval()
    sample = torch.randn(1, 4, 128, 128).to("cuda", torch.float32)
    timestep = torch.tensor([0], device="cuda")
    text_embeds = torch.randn(1, 1280).to("cuda")  # shape 
    time_ids = torch.randn(1, 6).to("cuda")         # shape = (B, 6)
    added_cond_kwargs = {
        "text_embeds": text_embeds,
        "time_ids": time_ids
    }

    encoder_hidden_states = torch.randn(1, 77, 2048).to("cuda")
    class UNetWrapper(torch.nn.Module):
        def __init__(self, model, timestep, encoder_hidden_states, added_cond_kwargs):
            super().__init__()
            self.model = model
            self.timestep = timestep
            self.encoder_hidden_states = encoder_hidden_states
            self.added_cond_kwargs = added_cond_kwargs

        def forward(self, sample):
            return self.model(sample, self.timestep, self.encoder_hidden_states, added_cond_kwargs=self.added_cond_kwargs).sample

    wrapped_model = UNetWrapper(model, timestep, encoder_hidden_states, added_cond_kwargs)
    flops, params = profile(wrapped_model, inputs=(sample,), verbose=False)
    print("UNet FLOPs:", flops / 1e12, "TFLOPs")
    # print(parameter_count_table(model))
    remove_thop_attributes(model)



flops_text()
flops_unet_singlestep()
flops_vae()
