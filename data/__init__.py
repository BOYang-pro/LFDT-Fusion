'''create dataset and dataloader'''
import logging



#================================================================
#                         load dataset
#=================================================================
def create_dataset_Fusion(dataset_opt, phase,opt):

    if opt['model']['fusion_task']=='VI-IR':           
        path1=dataset_opt['dataroot_ir']
        path2=dataset_opt['dataroot_vi']      
    elif opt['model']['fusion_task']=='VI-NIR':        
        path1=dataset_opt['dataroot_NIR']
        path2=dataset_opt['dataroot_Vis']
    elif opt['model']['fusion_task']=='PIF':           
        path1=dataset_opt['dataroot_DoLP']
        path2=dataset_opt['dataroot_S0']
    elif opt['model']['fusion_task']=='MEF':           
        path1=dataset_opt['dataroot_over']
        path2=dataset_opt['dataroot_under']
    elif opt['model']['fusion_task']=='MFF':           
        path1=dataset_opt['dataroot_source_1']
        path2=dataset_opt['dataroot_source_2']
    elif opt['model']['fusion_task']=='SPECT-MRI':     
        path1=dataset_opt['dataroot_SPECT_MRI']
        path2=dataset_opt['dataroot_SPECT']
    elif opt['model']['fusion_task']=='PET-MRI':
        path1=dataset_opt['dataroot_PET_MRI']
        path2=dataset_opt['dataroot_PET']
   
    
    
    if opt['model']['Fusion']['mode']=='MAX':
      from data.Fusion_dataset import FusionDataset
      '''create dataset'''
      dataset = FusionDataset(split=phase,                    
                  crop_size=dataset_opt['crop_size'],         
                  min_max=(-1,1),                            
                  ir_path=path1,                            # IR image path
                  vi_path=path2,                            # VI image path
                  is_crop=dataset_opt['is_crop'],           
                  )
    
    if opt['model']['Fusion']['mode']=='MEAN':
      from data.Fusion_dataset import FusionDataset_Digtal
      '''create dataset'''
      dataset = FusionDataset_Digtal(split=phase,          
                  crop_size=dataset_opt['crop_size'],         
                  min_max=(-1,1),                             
                  img1_path=path1,                 
                  img2_path=path2,                  
                  is_crop=dataset_opt['is_crop'],             
                  )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,dataset_opt['name']))
    return dataset
