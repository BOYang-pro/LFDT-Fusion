import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
logger = logging.getLogger('base')
##########################################
#                    Weight initialization
##########################################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


##########################################
#                define Network
##########################################


# Generator
def define_Network(opt,local_rank):
    from model.head.fusion_pipiline  import Fusion_Pipiline

    #latent space
    model_opt = opt['model']
    Unet_opt=model_opt['Unet']
    if Unet_opt['LDM']=='U_LDM': 
        from model.diffusers.UNet_arch import Encode,Decode
        LDM_Enc= Encode(in_ch=Unet_opt['in_ch'],
                          out_ch=Unet_opt['out_ch'],
                          ch=Unet_opt['LDM_ch'],
                          ch_mult=Unet_opt['ch_mult'],
                          embed_dim=Unet_opt['LDM_embed_dim'])
        LDM_Dec = Decode(in_ch=Unet_opt['in_ch'],
                          out_ch=Unet_opt['out_ch'],
                          ch=Unet_opt['LDM_ch'],
                          ch_mult=Unet_opt['ch_mult'],
                          embed_dim=Unet_opt['LDM_embed_dim'])
    
    # Diffusion fusion model
    Fusion_model = Fusion_Pipiline(
        LDM_Enc,
        LDM_Dec,
        mode=model_opt['Fusion']['mode'],
        fusion_task=opt['model']['fusion_task'],
        sample_selected=model_opt['Fusion']['sample_selected'],
        model_selected=model_opt['Fusion']['model_selected'],
        feat_channels = model_opt['Fusion']['feat_channels'],
        inference_steps = model_opt['Fusion']['inference_steps'],
        num_train_timesteps = model_opt['Fusion']['num_train_timesteps'],
        channel_emdin=model_opt['Fusion']['channel_emdin'],
        num_blocks=model_opt['Fusion']['num_blocks'],
        heads=model_opt['Fusion']['heads']
    )
    
    if opt['phase'] == 'train':
        load_path = opt['path']['resume_state']
        
        if load_path is None:
            init_weights(Fusion_model.model, init_type='normal')
            init_weights(Fusion_model.fusion, init_type='normal')
        

    #Determine whether to enable distributed multi-GPU computing
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        #BN layer synchronization
        Fusion_model = nn.SyncBatchNorm.convert_sync_batchnorm(Fusion_model)
        # Put a copy of the model into the DDP
        Fusion_model = nn.parallel.DistributedDataParallel(Fusion_model.cuda(local_rank),device_ids=[local_rank], find_unused_parameters=True)
    return Fusion_model
    
