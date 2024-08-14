# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from model.diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from model.diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from model.diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from model.diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from model.diffusers.schedulers.scheduling_pndm import PNDMScheduler

from typing import Union, Dict, Tuple, Optional
from mmengine.model import BaseModule
from model.loss import loss
from model.head.fusion_head import Fusion_head
from model.head.DFT_arch import DFT
class Fusion_Pipiline(BaseModule):
    def __init__(
            self,
            LDM_Enc,
            LDM_Dec,
            sample_selected,
            model_selected,
            feat_channels,                  
            inference_steps=5,           #Represents the number of inference steps used for diffusion steps in a diffusion model
            num_train_timesteps=1000,    #Represents the number of training steps, which specifies the total number of steps in the model training process
            mode='Max',
            fusion_task='MEF',
            channel_emdin=8,
            num_blocks=[4, 4, 4, 4],    
            heads=[1, 2, 4, 8],
            bias=False,
            LayerNorm_type='WithBias',
    ):
        super().__init__()
        self.mode=mode
        self.LDM_Enc=LDM_Enc
        self.LDM_Dec=LDM_Dec
        self.sample_selected=sample_selected
        self.fusion_task=fusion_task
        self.model_selected=model_selected
        if self.fusion_task=='MEF':
            ffn_factor=2.66
        else:
            ffn_factor=4.0
        # Sampling step
        self.diffusion_inference_steps = inference_steps

        #Denoising network 
        if self.model_selected=='DFT':
            self.model = DFT(inp_channels=channel_emdin*2,
                         out_channels=channel_emdin*2,
                         dim=feat_channels[0],
                         num_blocks=num_blocks,
                         heads=heads,
                         ffn_factor=ffn_factor,
                         bias=bias,
                         LayerNorm_type=LayerNorm_type,
                         num_channel=feat_channels)
                                                   
        # Fusion head
        self.fusion = Fusion_head(
                out_channels=8,
                dim=128,
                num_blocks=[2, 2, 2, 2],
                heads=[1, 2, 4, 8],
                ffn_factor=2.66,
                bias=False,
                LayerNorm_type='WithBias',
                )
    
        
        #Diffusion sampler
        if self.sample_selected =='DDIM':
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        if self.sample_selected =='ddp-solver':
            self.scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected =='ddp-solver++':
            self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected =='Deis':
            self.scheduler = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Unipc':
            self.scheduler = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'LMS':
            self.scheduler = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Heun':
            self.scheduler = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'PNDM':
            self.scheduler = PNDMScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Euler':
            self.scheduler = EulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)

        #Iterative denoising framework
        self.pipeline = DenoisePipiline(self.model, self.scheduler, self.sample_selected)

    def set_loss(self,device):
        #Diffusion loss
        self.dif_loss = nn.MSELoss().to(device)
        #Fusion loss
        if self.fusion_task=='MEF' or self.fusion_task=='MFF':
            self.fusion_loss = loss.Fusion_loss(mode=self.mode,lambda1=10,lambda2=20,lambda3=20).to(device)
        else:
            self.fusion_loss = loss.Fusion_loss(mode=self.mode,lambda1=10,lambda2=40,lambda3=40).to(device)
        
        

    def test_Fusion(self,x_in,device):
        x_vis = x_in[:, :1]
        x_ir = x_in[:, 1:]
        with torch.no_grad():
            batch_size = x_vis.shape[0]
            device = x_vis.device
            dtype = x_vis.dtype
            x1,x2,h= self.LDM_Enc(x_vis, x_ir)
            latent = torch.cat((x1,x2), dim=1)
            #Denoising process
            latent_result,_,_,middle_feat= self.pipeline(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                image=latent,  
                num_inference_steps=self.diffusion_inference_steps,  # 扩散步骤
                return_dict=False)
            Fusion_result=self.fusion(middle_feat,latent_result)
            Fusion_result= self.LDM_Dec(Fusion_result,h) 
            
        return Fusion_result
          
    def forward(self, image_vis, image_ir):
        
        batch_size = image_vis.shape[0]
        device = image_vis.device
        dtype=image_vis.dtype
        x1,x2,h= self.LDM_Enc(image_vis, image_ir)

        latent = torch.cat((x1,x2), dim=1)
        #Denoising process
        latent_result, latent_noise,pre_noise,middle_feat= self.pipeline(
                                                  batch_size=batch_size,
                                                  device=device,
                                                  dtype=dtype,
                                                  image=latent, 
                                                  num_inference_steps=self.diffusion_inference_steps, 
                                                  return_dict=False)
        
        #Loss calculation
        dif_loss = self.dif_loss(pre_noise,latent_noise)
        Fusion_result=self.fusion(middle_feat,latent_result)
        Fusion_result= self.LDM_Dec(Fusion_result,h)
        
   
        fusion_loss, loss_gradient, loss_l1, loss_SSIM=self.fusion_loss(image_vis, image_ir, Fusion_result)
        
        
        
        loss =fusion_loss+dif_loss
        output = {
                  'Fusion': Fusion_result,
                  'loss':loss,
                  'loss_gradient': loss_gradient,
                  'loss_l1': loss_l1,
                  'loss_SSIM': loss_SSIM,
                  'dif_loss': dif_loss
                  }
        return output
#D_t->D_t-1->D_t-2->....D_0

class DenoisePipiline:
    def __init__(self, model, scheduler,sample_selected):
        super().__init__()
        self.model = model
        self.scheduler = scheduler  
        self.sample_selected=sample_selected
    def __call__(
            self,
            batch_size,
            device,         
            dtype,           
            image,            
            generator: Optional[torch.Generator] = None, #Used to specify a random number generator that generates random numbers. If this parameter is not provided, the default random number generator is used
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,

    ) -> Union[Dict, Tuple]:
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError("generator.device == 'cpu'","0.11.0",message)
            generator = None

        # Set the sampling time
        self.scheduler.set_timesteps(num_inference_steps)
        # Generate random noise
        noise = torch.randn(image.shape).to(device)

        if self.sample_selected == 'DDIM':
            timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)  # Randomly generate a time step
            image = self.scheduler.add_noise(image, noise, timesteps).to(device) # Add noise
            noise_image = image
            for t in self.scheduler.timesteps:
                # Prediction model noise and retain intermediate features
                model_output,middle_feat = self.model(image, t, device)
                # do D_t->D_t-1->D_t-2->....D_0
                image = self.scheduler.step(model_output, t, image, eta=eta, use_clipped_model_output=True,generator=generator)['prev_sample']

        if self.sample_selected == 'ddp-solver' or self.sample_selected == 'ddp-solver++'or self.sample_selected == 'Deis' \
                or self.sample_selected == 'Unipc' or self.sample_selected == 'PNDM':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device) # Generate a time series
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)   # Add noise
            noise_image = image
          
            for t in self.scheduler.timesteps:
                # Prediction model noise and retain intermediate features
                model_output,middle_feat = self.model(image, t, device)
                # do D_t->D_t-1->D_t-2->....D_0
                image = self.scheduler.step(model_output, t, image)['prev_sample']
        if  self.sample_selected == 'Heun':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device) # Generate a time series
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)   # Add noise
            noise_image=image
            for t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                # Prediction model noise and retain intermediate features
                model_output,middle_feat = self.model(image, t, device)
                # do D_t->D_t-1->D_t-2->....D_0
                image = self.scheduler.step(model_output, t, image)['prev_sample']
        if self.sample_selected == 'LMS':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)  # Generate a time series

            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)  # Add noise
            noise_image = image
            #逐渐遍历时间步骤
            for t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                # Prediction model noise and retain intermediate features
                model_output,middle_feat = self.model(image, t, device)
                # do D_t->D_t-1->D_t-2->....D_0
                image = self.scheduler.step(model_output, t, image)['prev_sample']
        if self.sample_selected == 'Euler':
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)  # Generate a time series
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)  # Add noise
            noise_image = image
            generator = torch.manual_seed(0)

            for  t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                # Prediction model noise and retain intermediate features
                model_output,middle_feat = self.model(image, t, device)
                # do D_t->D_t-1->D_t-2->....D_0
                image = self.scheduler.step(model_output, t, image, generator=generator)['prev_sample']
        
        if not return_dict:
            return (image,noise_image,model_output,middle_feat)

        return {'images': image,'noise_image':noise_image}





