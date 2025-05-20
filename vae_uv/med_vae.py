
import torch
from pytorch_msssim import SSIM, ssim
import torch.nn as nn
import lpips
import torch.nn.functional as F
from utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl 
    
class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS)"""
    def __init__(self, linear_calibration=False, normalize=False):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net='vgg', lpips=linear_calibration) # Note: only 'vgg' valid as loss  
        self.normalize = normalize # If true, normalize [0, 1] to [-1, 1]
        

    def forward(self, pred, target):
        # No need to do that because ScalingLayer was introduced in version 0.1 which does this indirectly  
        # if pred.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB
        #     pred = torch.concat([pred, pred, pred], dim=1)
        # if target.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB 
        #     target = torch.concat([target, target, target], dim=1)

        if pred.ndim == 5: # 3D Image: Just use 2D model and compute average over slices 
            depth = pred.shape[2] 
            losses = torch.stack([self.loss_fn(pred[:,:,d], target[:,:,d], normalize=self.normalize) for d in range(depth)], dim=2)
            return torch.mean(losses, dim=2, keepdim=True)
        else:
            return self.loss_fn(pred, target, normalize=self.normalize)
        
class VAE(nn.Module):
    def __init__(
        self,
        in_channels=3, 
        out_channels=3, 
        spatial_dims = 2,
        emb_channels = 4,
        hid_chs =    [ 64, 128,  256, 512],
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name=("Swish", {}),
        dropout=None,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=True,
        use_attention='none',
        embedding_loss_weight=1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        

        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        loss = torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},

        sample_every_n_steps = 1000

    ):
        super(VAE, self).__init__()
        # super().__init__(
        #     optimizer=optimizer,
        #     optimizer_kwargs=optimizer_kwargs,
        #     lr_scheduler=lr_scheduler,
        #     lr_scheduler_kwargs=lr_scheduler_kwargs
        # )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        # self.ssim_fct = SSIM(data_range=1, size_average=False, channel=out_channels, spatial_dims=spatial_dims, nonnegative_ssim=True)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        # -------- Loss-Reg---------
        # self.logvar = nn.Parameter(torch.zeros(size=()) )

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name, 
            norm_name=norm_name,
            emb_channels=None
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = norm_name,
                act_name = act_name,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = None
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2*emb_channels, 3),
            BasicBlock(spatial_dims, 2*emb_channels, 2*emb_channels, 1)
        )


        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()    


        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])
        # self.logvar_ver = nn.ParameterList([
        #     nn.Parameter(torch.zeros(size=()) )
        #     for _ in range(1, deep_supervision+1)
        # ])
        self.u_filter = nn.Conv2d(2, 2, 3, 1, 1)
        self.v_filter = nn.Conv2d(2, 2, 3, 1, 1)
        self.c_filter = nn.Conv2d(2*2, 2, 3, 1, 1)
    
    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z 
            
    def decode(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x 

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.inc(x_in)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        # --------- Quantizer --------------
        #z_q, emb_loss = self.quantizer(z)
        z_q, emb_loss = self.quantizer(z)

        z1, z2 = torch.split(z_q, split_size_or_sections=[z_q.size(1) // 2, z_q.size(1) // 2], dim=1)
        u = self.u_filter(z1)
        v = self.v_filter(z2)
        c = self.c_filter(torch.cat((z1,z2),dim=1))

        z1 = u + c
        z2 = v + c
        z_re = torch.cat((z1,z2),dim=1)
        #z_re = z_q

        # -------- Decoder -----------
        out_hor = []
        #h = self.inc_dec(z_q)
        h = self.inc_dec(z_re)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)
   
        return out, out_hor[::-1], emb_loss, u, v, c
        #return out, out_hor[::-1], emb_loss
    
    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth<2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0 
    
    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False, 
                        nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))
    
    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0

        outputs1 = pred[:, :1, :, :]
        images1 = target[:, :1, :, :]

        outputs2 = pred[:, 1:, :, :]
        images2 = target[:, 1:, :, :]
        self.lpips1 = self.perception_loss(outputs1, images1)
        self.lpips2 = self.perception_loss(outputs2, images2)
        rec_loss = self.loss_fct(pred, target) + self.lpips1 + self.lpips2 + self.ssim_loss(pred, target)

        # rec_loss = self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)
        # rec_loss = rec_loss/ torch.exp(self.logvar) + self.logvar # Note this is include in Stable-Diffusion but logvar is not used in optimizer 
        loss += torch.sum(rec_loss)/pred.shape[0]  
        

        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            outputs1 = pred_i[:, :1, :, :]
            images1 = target_i[:, :1, :, :]

            outputs2 = pred_i[:, 1:, :, :]
            images2 = target_i[:, 1:, :, :]
            self.lpips1 = self.perception_loss(outputs1, images1)
            self.lpips2 = self.perception_loss(outputs2, images2)
            rec_loss_i = self.loss_fct(pred_i, target_i)+self.lpips1+self.lpips2+self.ssim_loss(pred_i, target_i)
            #rec_loss_i = self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i] 
            loss += torch.sum(rec_loss_i)/pred.shape[0]  

        return loss 
