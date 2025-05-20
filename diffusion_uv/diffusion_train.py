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
latent_embedder_checkpoint = '/data01/home/yangsc/med_diff_20250514/vae_uv/pth/vae_model_50.pth'

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
from tqdm import tqdm
import torch.optim as optim
# import torch
# from dataset import brain_train_paired
from knee_dataset import knee_train_paired
from torch.utils.data import DataLoader
num_epochs = 100
# train_dataset = brain_train_paired(t1_dir='/data01/home/yangsc/med_uc_diff/LGG_data_skip50/T1n_test',t2_dir='/data01/home/yangsc/med_uc_diff/LGG_data_skip50/T2w_test')
t1_dir = '/data01/home/yangsc/FMRI/process_data/t1_o'
t2_dir = '/data01/home/yangsc/FMRI/process_data/t2_o'
train_dataset = knee_train_paired(t1_dir=t1_dir,t2_dir=t2_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# optimizer = optim.Adam(model.parameters(), lr=8e-5)

optimizer = torch.optim.AdamW(params=model.parameters(), **{'lr': 1e-5, 'weight_decay': 1e-4})
iter_num = 0
for epoch in range(num_epochs):
    total_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch in pbar :
        t1 = batch['t1'].cuda()
        t2 = batch['t2'].cuda()
        images = torch.cat((t1,t2),dim=1)
        # vae.eval()
        # zq = vae.encode(images)
        loss = model._step(images)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        total_loss += loss.item()
        average_loss = total_loss / (pbar.n + 1)  
        pbar.set_postfix(loss=average_loss,epoch=epoch+1)  
        iter_num+=1
        # if iter_num>0:
        if iter_num%500==0:
            sample_cond = None
            sample_img = model.sample(num_samples=1, img_size=[4,60,32], condition=sample_cond).detach()
            # img_path = '/data01/home/yangsc/med_uc_diff/diffusion_module/vis'
            # os.mkdir(img_path,exist_ok=True)  
            logits1_np = sample_img[0][0].detach().cpu().numpy().clip(0,1)
            logits2_np = sample_img[0][1].detach().cpu().numpy().clip(0,1)
            save_image(logits1_np, f'ddpm_output/{epoch+1}_{iter_num}_t1.png')
            save_image(logits2_np, f'ddpm_output/{epoch+1}_{iter_num}_t2.png') 
    torch.save(model.state_dict(), f'./ddpm_pth/diffusion_model_{epoch}.pth')
        # iter_num+=1
        # if iter_num>0:
        #     sample_cond = None
        #     sample_img = model.sample(num_samples=model.num_samples, img_size=[4,32,32], condition=sample_cond).detach()
        #     print()
        # if iter%300 == 0:
        #     sampled_z = diffusion.sample(batch_size = 1)
        #     with torch.no_grad():
        #         z1,z2 = torch.split(sampled_z, [sampled_z.shape[1]//2, sampled_z.shape[1]//2], dim=1)
        #         pred_u = vae.u_filter(z1)
        #         pred_v = vae.v_filter(z2)
        #         pred_c = vae.c_filter(sampled_z)
        #         pred_z1 = pred_u + pred_c
        #         pred_z2 = pred_v + pred_c
        #         z_re = torch.cat((pred_z1,pred_z2),dim=1)
        #         out = vae.decode(z_re)
        #         logits1_np = out[0][0].detach().cpu().numpy().clip(0,1)
        #         logits2_np = out[0][1].detach().cpu().numpy().clip(0,1)
        #         save_image(logits1_np, f'ddpm_output_hou/images/{epoch}_{iter}_t1.png')
        #         save_image(logits2_np, f'ddpm_output_hou/images/{epoch}_{iter}_t2.png')