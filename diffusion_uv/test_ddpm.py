from diffusion_model import DiffusionPipeline
from model_utils import TimeEmbbeding, UNet, GaussianNoiseScheduler
# cond_embedder = LabelEmbedder
# cond_embedder_kwargs = {
#     'emb_dim': 1024,
#     'num_classes': 2
# }
from med_vae import VAE
time_embedder = TimeEmbbeding
time_embedder_kwargs ={
    'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
}
import torch
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # 如果使用GPU的话
noise_estimator = UNet
noise_estimator_kwargs = {
    'in_ch':4, 
    'out_ch':4, 
    'spatial_dims':2,
    'hid_chs':  [  256, 256, 512, 1024],
    'kernel_sizes':[3, 3, 3, 3],
    'strides':     [1, 2, 2, 2],
    'time_embedder':time_embedder,
    'time_embedder_kwargs': time_embedder_kwargs,
    'cond_embedder':None,
    'cond_embedder_kwargs': None,
    'deep_supervision': False,
    'use_res_block':True,
    'use_attention':'none',
}


# ------------ Initialize Noise ------------
noise_scheduler = GaussianNoiseScheduler
noise_scheduler_kwargs = {
    'timesteps': 1000,
    'beta_start': 0.002, # 0.0001, 0.0015
    'beta_end': 0.02, # 0.01, 0.0195
    'schedule_strategy': 'scaled_linear'
}


latent_embedder = VAE(in_channels=2,out_channels=2).cuda()
latent_embedder_checkpoint = '/data01/home/yangsc/med_uc_diff/out_vae_t1nt2w/vae_model_49.pth'

model = DiffusionPipeline(
    noise_estimator=noise_estimator, 
    noise_estimator_kwargs=noise_estimator_kwargs,
    noise_scheduler=noise_scheduler, 
    noise_scheduler_kwargs = noise_scheduler_kwargs,
    latent_embedder=latent_embedder,
    latent_embedder_checkpoint = latent_embedder_checkpoint,
    estimator_objective='x_T',
    estimate_variance=False, 
    use_self_conditioning=False, 
    use_ema=False,
    classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
    do_input_centering=False,
    clip_x0=False,
    sample_every_n_steps=1000
).cuda()



# import torch
# a = torch.randn(1,2,256,256).cuda()
# batch = {'source':a}
# loss = model._step(batch)
# print(loss.item())
# print('ok')

import os
from PIL import Image
def save_image(image_array, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_array = image_array * 255.0
    image = Image.fromarray(image_array.astype('uint8'))
    image.save(filename)

model_path = '/data01/home/yangsc/med_uc_diff/diffusion_module/pth/diffusion_model_99.pth'
model.load_state_dict(torch.load(model_path))
num_epochs = 1000
model.eval()
from tqdm import tqdm
for epoch in tqdm(range(num_epochs)):
    sample_cond = None
    with torch.no_grad():
        sample_img = model.sample(num_samples=1, img_size=[4,32,32], condition=sample_cond).detach()
    logits1_np = sample_img[0][0].detach().cpu().numpy().clip(0,1)
    logits2_np = sample_img[0][1].detach().cpu().numpy().clip(0,1)
    save_image(logits1_np, f'ddpm_output_v3/sample_images/{epoch+1}_sample_t1.png')
    save_image(logits2_np, f'ddpm_output_v3/sample_images/{epoch+1}_sample_t2.png') 