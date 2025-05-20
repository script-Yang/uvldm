# from unet_model import UNet
# from our_model import UNet
import torch
import lpips  
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from dataset import brain_train_paired
from knee_dataset import knee_train_paired
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
# from pytorch_msssim import SSIM, ssim
import imageio
from PIL import Image
# from res_model import Generator
# from vae import AutoEncoderKL
from med_vae import VAE
#model = UNet(n_channels=1, n_classes=1, bilinear='bilinear').cuda()
#model = Generator(input_nc=1,output_nc=1,n_residual_blocks=4).cuda()
# model = AutoEncoderKL(num_channels=2,latent_dim=128).cuda()

model = VAE(in_channels=2,out_channels=2).cuda()
#model = VAE(in_channels=2,out_channels=1).cuda()

mse_loss = nn.MSELoss().cuda()
lpips_loss = lpips.LPIPS(net='vgg').cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
t1_dir = '/data01/home/yangsc/FMRI/process_data/t1_o'
t2_dir = '/data01/home/yangsc/FMRI/process_data/t2_o'
train_dataset = knee_train_paired(t1_dir=t1_dir,t2_dir=t2_dir)
#train_dataset = brain_train_paired(t1_dir='/data01/home/yangsc/med_uc_diff/LGG_data_skip50/T1n_test',t2_dir='/data01/home/yangsc/med_uc_diff/LGG_data_skip50/T2w_test')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

def save_image(image_array, filename):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    image_array = image_array * 255.0
    image = Image.fromarray(image_array.astype('uint8'))
    image.save(filename)

num_epochs = 50
iter = 0 
for epoch in range(num_epochs):
    total_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch in pbar :
        t1 = batch['t1'].cuda()
        t2 = batch['t2'].cuda()
        #images = torch.cat((t1,t2),dim=0)
        images = torch.cat((t1,t2),dim=1)
        #outputs,z1,z2,x5_1,x5_2,x5_1_recon,x5_2_recon = model(images)
        
        #outputs,_,_,_ = model(images)
        #outputs,_,_ = model(images)
        outputs, pred_vertical, emb_loss, _, _, _  = model(images)
        loss = model.rec_loss(outputs, pred_vertical, images)
        loss += emb_loss*model.embedding_loss_weight
        #loss = loss_mse

        optimizer.zero_grad() 
        loss.backward()  
        optimizer.step()      

        total_loss += loss.item()
        average_loss = total_loss / (pbar.n + 1)  
        pbar.set_postfix(loss=average_loss,epoch=epoch+1)  
        iter+=1
        if iter%100 == 0:
            img_np = outputs[0][0].detach().cpu().numpy().clip(0,1)
            gt_np = images[0][0].detach().cpu().numpy()

            psnr_value = psnr(img_np,gt_np,data_range=1)
            ssim_value = ssim(img_np,gt_np,data_range=1)
            
            print(f'PSNR: {psnr_value:.4f} dB')
            print(f'SSIM: {ssim_value:.4f}')
            save_image(img_np, f'./images/{epoch+1}_{iter}_pred.png')
            save_image(gt_np, f'./images/{epoch+1}_{iter}_gt.png')
    torch.save(model.state_dict(), f'./pth/vae_model_{epoch+1}.pth')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
# print('ok')
