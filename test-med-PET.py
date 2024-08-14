import os
import torch
import model as Model
import argparse
import core.logger as Logger
import os
import numpy as np
from util.util  import  RGB2YCrCb,YCrCb2RGB
from torchvision.transforms import ToTensor
from PIL import Image
from util.img_read_save import img_save
from tqdm import tqdm
import time
os.environ["CUDA_PET_DEVICES"] = "1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_med-PET.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    #=====================================================================
    #           Parsing command
    #======================================================================
    
    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    for dataset_name in ["PET-MRI"]:
            print("The test result of " + dataset_name + ' :')
            test_folder = os.path.join('./dataset/test/Med/', dataset_name)
            test_out_folder = os.path.join('./dataset/test_result/MED/', dataset_name+'/')
            if not os.path.exists(test_out_folder):
                os.makedirs(test_out_folder)
            device = torch.device(f'cuda' if opt['gpu_ids'] is not None else 'cpu')
            diffusion = Model.create_model(opt, args.local_rank)
            
            diffusion.Fusion_net.to(device)
            diffusion.Fusion_net.eval()
                
            #计算DDPM的参数量
            total = sum([param.nelement() for param in diffusion.Fusion_net.parameters()])
            print("Number of parameters: %.2fM" % (total / 1e6))
            
            min_max = (-1, 1)
            with torch.no_grad():
                for img_name in tqdm(os.listdir(os.path.join(test_folder, "MRI"))):
                        
                    PET_image = Image.open(os.path.join(test_folder, "PET", img_name)).convert('RGB')
                    MRI_image = Image.open(os.path.join(test_folder, "MRI", img_name))

                    PET_image = (ToTensor()(PET_image) * (min_max[1] - min_max[0]) +min_max[0]).unsqueeze(0).cuda()
                    MRI_image = (ToTensor()(MRI_image) * (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0).cuda()

                    PET_image_YUV = RGB2YCrCb(PET_image)
                    data_PET=PET_image_YUV[:, 0:1, :, :]
                    data_MRI=MRI_image[:, 0:1, :, :]
                    input = torch.cat([data_PET, data_MRI], dim=1)
                    output=diffusion.Fusion_net.test_Fusion(input, device)
                    YUV= YCrCb2RGB(torch.cat((output, PET_image_YUV[:, 1:2, :, :], PET_image_YUV[:, 2:3, :, :]), dim=1))
                    RGB = diffusion.tensor2fu(YUV, min_max=(0, 1)).astype(np.uint8)
                    output=diffusion.tensor2fu(output, min_max=(0, 1)).astype(np.uint8)
                    img_save(RGB, img_name.split(sep='.')[0], test_out_folder)
                        