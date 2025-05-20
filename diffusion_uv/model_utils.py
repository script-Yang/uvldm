import math 
import torch 
import torch.nn as nn 
from monai.networks.layers.utils import get_act_layer
from monai.networks.blocks import UnetOutBlock
from utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from utils.conv_blocks import BasicBlock, UpBlock, DownBlock, UnetBasicBlock, UnetResBlock, save_add, BasicDown, BasicUp, SequentialEmb
from utils.attention_blocks import Attention, zero_module

class SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim=16, downscale_freq_shift=1, max_period=10000, flip_sin_to_cos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos=flip_sin_to_cos

    def forward(self, x):
        device = x.device
        half_dim = self.emb_dim // 2
        emb = math.log(self.max_period) / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb*torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        
        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        half_dim = emb_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        if self.emb_dim % 2 == 1:
            fouriered = torch.nn.functional.pad(fouriered, (0, 1, 0, 0))
        return fouriered



class TimeEmbbeding(nn.Module):
    def __init__(
            self, 
            emb_dim = 64,  
            pos_embedder = SinusoidalPosEmb,
            pos_embedder_kwargs = {},
            act_name=("SWISH", {}) # Swish = SiLU 
        ):
        super().__init__()
        self.emb_dim = emb_dim
        self.pos_emb_dim =  pos_embedder_kwargs.get('emb_dim', emb_dim//4)
        pos_embedder_kwargs['emb_dim'] = self.pos_emb_dim
        self.pos_embedder = pos_embedder(**pos_embedder_kwargs)
        

        self.time_emb = nn.Sequential(
            self.pos_embedder,
            nn.Linear(self.pos_emb_dim, self.emb_dim),
            get_act_layer(act_name),
            nn.Linear(self.emb_dim, self.emb_dim)
        )
    
    def forward(self, time):
        return self.time_emb(time)
    


class UNet(nn.Module):

    def __init__(self, 
            in_ch=1, 
            out_ch=1, 
            spatial_dims = 3,
            hid_chs =    [256, 256, 512,  1024],
            kernel_sizes=[ 3,  3,   3,   3],
            strides =    [ 1,  2,   2,   2], # WARNING, last stride is ignored (follows OpenAI)
            act_name=("SWISH", {}),
            norm_name = ("GROUP", {'num_groups':32, "affine": True}),
            time_embedder=TimeEmbbeding,
            time_embedder_kwargs={},
            cond_embedder=None,
            cond_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning = False, 
            dropout=0.0, 
            learnable_interpolation=True,
            use_attention='none',
            num_res_blocks=2,
        ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        self.num_res_blocks = num_res_blocks

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None 

        # ------------- Condition-Embedder-----------
        if cond_embedder is not None:
            self.cond_embedder=cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None 
            cond_emb_dim = None 


        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # ----------- In-Convolution ------------
        in_ch = in_ch*2 if self.use_self_conditioning else in_ch 
        self.in_conv = BasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0])
        
        
        # ----------- Encoder ------------
        in_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = [] 
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i-1 if k==0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )

                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i]//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))

            if i < self.depth-1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation 
                    )
                )
 

        self.in_blocks = nn.ModuleList(in_blocks)
        
        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1]//8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=time_emb_dim,
                attention_type=use_attention[-1]
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            )
        )

 
     
        # ------------ Decoder ----------
        out_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks+1):
                seq_list = [] 
                out_channels=hid_chs[i-1 if k==0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i]+hid_chs[i-1 if k==0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
            
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )

                if (i >1) and k==0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation 
                        )
                    )
        
                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)
        
        
        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = zero_module(UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None))
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-2 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            zero_module(UnetOutBlock(spatial_dims, hid_chs[i]+hid_chs[i-1], out_ch, dropout=None) )
            for i in range(2, deep_supervision+2)
        ])
 

    def forward(self, x_t, t=None, condition=None, self_cond=None):
        # x_t [B, C, *]
        # t [B,]
        # condition [B,]
        # self_cond [B, C, *]
        

        # -------- Time Embedding (Gloabl) -----------
        if t is None:
            time_emb = None 
        else:
            time_emb = self.time_embedder(t) # [B, C]

        # -------- Condition Embedding (Gloabl) -----------
        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None  
        else:
            cond_emb = self.cond_embedder(condition) # [B, C]
        
        emb = save_add(time_emb, cond_emb)
       
        # ---------- Self-conditioning-----------
        if self.use_self_conditioning:
            self_cond =  torch.zeros_like(x_t) if self_cond is None else x_t 
            x_t = torch.cat([x_t, self_cond], dim=1)  
    
        # --------- Encoder --------------
        x = [self.in_conv(x_t)]
        for i in range(len(self.in_blocks)):
            x.append(self.in_blocks[i](x[i], emb))

        # ---------- Middle --------------
        h = self.middle_block(x[-1], emb)
        
        # -------- Decoder -----------
        y_ver = []
        for i in range(len(self.out_blocks), 0, -1):
            h = torch.cat([h, x.pop()], dim=1)

            depth, j = i//(self.num_res_blocks+1), i%(self.num_res_blocks+1)-1
            y_ver.append(self.outc_ver[depth-1](h)) if (len(self.outc_ver)>=depth>0) and (j==0) else None 

            h = self.out_blocks[i-1](h, emb)

        # ---------Out-Convolution ------------
        y = self.outc(h)

        return y, y_ver[::-1]




# if __name__=='__main__':
#     model = UNet(in_ch=3, use_res_block=False, learnable_interpolation=False)
#     input = torch.randn((1,3,16,32,32))
#     time = torch.randn((1,))
#     out_hor, out_ver = model(input, time)
#     print(out_hor[0].shape)



import torch 
import torch.nn.functional as F
class BasicNoiseScheduler(nn.Module):
    def __init__(
        self,
        timesteps=1000,
        T=None,
        ):
        super().__init__()
        self.timesteps = timesteps 
        self.T = timesteps if  T is None else T 

        self.register_buffer('timesteps_array', torch.linspace(0, self.T-1, self.timesteps, dtype=torch.long))   # NOTE: End is inclusive therefore use -1 to get [0, T-1]  
    
    def __len__(self):
        return len(self.timesteps)

    def sample(self, x_0):
        """Randomly sample t from [0,T] and return x_t and x_T based on x_0"""
        t = torch.randint(0, self.T, (x_0.shape[0],), dtype=torch.long, device=x_0.device) # NOTE: High is exclusive, therefore [0, T-1]
        x_T = self.x_final(x_0) 
        return self.estimate_x_t(x_0, t, x_T), x_T, t
    
    def estimate_x_t_prior_from_x_T(self, x_T, t, **kwargs):
        raise NotImplemented
    
    def estimate_x_t_prior_from_x_0(self, x_0, t, **kwargs):
        raise NotImplemented

    def estimate_x_t(self, x_0, t, x_T=None, **kwargs):
        """Get x_t at time t"""
        raise NotImplemented

    @classmethod
    def x_final(cls, x):
        """Get noise that should be obtained for t->T """
        raise NotImplemented

    @staticmethod 
    def extract(x, t, ndim):
        """Extract values from x at t and reshape them to n-dim tensor"""
        return x.gather(0, t).reshape(-1, *((1,)*(ndim-1))) 

#from medical_diffusion.models.noise_schedulers import BasicNoiseScheduler

class GaussianNoiseScheduler(BasicNoiseScheduler):
    def __init__(
        self,
        timesteps=1000,
        T = None, 
        schedule_strategy='cosine',
        beta_start = 0.0001, # default 1e-4, stable-diffusion ~ 1e-3
        beta_end = 0.02,
        betas = None,
        ):
        super().__init__(timesteps, T)

        self.schedule_strategy = schedule_strategy

        if betas is not None:
            betas = torch.as_tensor(betas, dtype = torch.float64)
        elif schedule_strategy == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
        elif schedule_strategy == "scaled_linear": # proposed as "quadratic" in https://arxiv.org/abs/2006.11239, used in stable-diffusion 
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64)**2
        elif schedule_strategy == "cosine":
            s = 0.008
            x = torch.linspace(0, timesteps, timesteps + 1, dtype = torch.float64) # [0, T]
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas =  torch.clip(betas, 0, 0.999)
        else:
            raise NotImplementedError(f"{schedule_strategy} does is not implemented for {self.__class__}")


        alphas = 1-betas 
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)


        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas) # (0 , 1)

        register_buffer('alphas', alphas) # (1 , 0)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',  torch.sqrt(1. / alphas_cumprod - 1))

        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
            

    def estimate_x_t(self, x_0, t, x_T=None):
        # NOTE: t == 0 means diffused for 1 step (https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py#L108)
        # NOTE: t == 0 means not diffused for cold-diffusion (in contradiction to the above comment) https://github.com/arpitbansal297/Cold-Diffusion-Models/blob/c828140b7047ca22f995b99fbcda360bc30fc25d/denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L361
        x_T = self.x_final(x_0) if x_T is None else x_T 
        # ndim = x_0.ndim
        # x_t = (self.extract(self.sqrt_alphas_cumprod, t, ndim)*x_0 + 
        #         self.extract(self.sqrt_one_minus_alphas_cumprod, t, ndim)*x_T)
        def clipper(b):
            tb = t[b]
            if tb<0:
                return x_0[b]
            elif tb>=self.T:
                return x_T[b] 
            else:
                return self.sqrt_alphas_cumprod[tb]*x_0[b]+self.sqrt_one_minus_alphas_cumprod[tb]*x_T[b]
        x_t = torch.stack([clipper(b) for b in range(t.shape[0])]) 
        return x_t 
    

    def estimate_x_t_prior_from_x_T(self, x_t, t, x_T, use_log=True, clip_x0=True,  var_scale=0, cold_diffusion=False): 
        x_0 = self.estimate_x_0(x_t, x_T, t, clip_x0)
        return self.estimate_x_t_prior_from_x_0(x_t, t, x_0, use_log, clip_x0, var_scale, cold_diffusion)


    def estimate_x_t_prior_from_x_0(self, x_t, t, x_0, use_log=True, clip_x0=True, var_scale=0, cold_diffusion=False):
        x_0 = self._clip_x_0(x_0) if clip_x0 else x_0

        if cold_diffusion: # see https://arxiv.org/abs/2208.09392 
            x_T_est =  self.estimate_x_T(x_t, x_0, t) # or use x_T estimated by UNet if available? 
            x_t_est = self.estimate_x_t(x_0, t, x_T=x_T_est) 
            x_t_prior = self.estimate_x_t(x_0, t-1, x_T=x_T_est) 
            noise_t = x_t_est-x_t_prior
            x_t_prior = x_t-noise_t 
        else:
            mean = self.estimate_mean_t(x_t, x_0, t)
            variance = self.estimate_variance_t(t, x_t.ndim, use_log, var_scale)
            std = torch.exp(0.5*variance) if use_log else torch.sqrt(variance)
            std[t==0] = 0.0 
            x_T = self.x_final(x_t)
            x_t_prior =  mean+std*x_T
        return x_t_prior, x_0 

    
    def estimate_mean_t(self, x_t, x_0, t):
        ndim = x_t.ndim
        return (self.extract(self.posterior_mean_coef1, t, ndim)*x_0+
                self.extract(self.posterior_mean_coef2, t, ndim)*x_t) 
    

    def estimate_variance_t(self, t, ndim, log=True, var_scale=0, eps=1e-20):
        min_variance = self.extract(self.posterior_variance, t, ndim)
        max_variance = self.extract(self.betas, t, ndim)
        if log:
            min_variance = torch.log(min_variance.clamp(min=eps))
            max_variance = torch.log(max_variance.clamp(min=eps))
        return var_scale * max_variance + (1 - var_scale) * min_variance 
    

    def estimate_x_0(self, x_t, x_T, t, clip_x0=True):
        ndim = x_t.ndim
        x_0 = (self.extract(self.sqrt_recip_alphas_cumprod, t, ndim)*x_t - 
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, ndim)*x_T)
        x_0 = self._clip_x_0(x_0) if clip_x0 else x_0
        return x_0


    def estimate_x_T(self, x_t, x_0, t, clip_x0=True):
        ndim = x_t.ndim
        x_0 = self._clip_x_0(x_0) if clip_x0 else x_0
        return ((self.extract(self.sqrt_recip_alphas_cumprod, t, ndim)*x_t - x_0)/ 
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, ndim))
    
    
    @classmethod
    def x_final(cls, x):
        return torch.randn_like(x)

    @classmethod
    def _clip_x_0(cls, x_0):
        # See "static/dynamic thresholding" in Imagen https://arxiv.org/abs/2205.11487 

        # "static thresholding"
        m = 1 # Set this to about 4*sigma = 4 if latent diffusion is used  
        x_0 = x_0.clamp(-m, m)
        return x_0