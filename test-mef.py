import os

import torch
import model as Model
import argparse
import core.logger as Logger
import os
import numpy as np
from util.util  import  RGB2YCrCb
from torchvision.transforms import ToTensor
from PIL import Image
from util.img_read_save import img_save
from tqdm import tqdm

os.environ["CUDA_over_DEVICES"] = "2"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_mef.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    #=====================================================================
    #           Parsing command
    #======================================================================

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    for dataset_name in ["MEF"]:
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('./dataset/test/MEF/', dataset_name)
        test_out_folder = os.path.join('./dataset/test_result/MEF/MEF/', dataset_name+'/')
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)

                # 实例化模型
        diffusion = Model.create_model(opt, args.local_rank)

        device = torch.device(f'cuda' if opt['gpu_ids'] is not None else 'cpu')
        diffusion.Fusion_net.to(device)
        diffusion.Fusion_net.eval()
        
        min_max = (-1, 1)
        with torch.no_grad():
            for img_name in tqdm(os.listdir(os.path.join(test_folder, "over"))):
                over_image = Image.open(os.path.join(test_folder, "over", img_name)).convert('RGB')
                under_image = Image.open(os.path.join(test_folder, "under", img_name)).convert('RGB')

                # 将图像转换为PyTorch张量并将其缩放到指定大小[-1，1]
                over_image = (ToTensor()(over_image) * (min_max[1] - min_max[0]) +min_max[0]).unsqueeze(0).cuda()
                under_image = (ToTensor()(under_image) * (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0).cuda()
                        
                [b,c,m,n]=over_image.shape
                       
                            
                over_image_YUV = RGB2YCrCb(over_image)
                under_image_YUV = RGB2YCrCb(under_image)
                     
                data_over=over_image_YUV[:, 0:1, :, :]
                data_under=under_image_YUV[:, 0:1, :, :]

                input = torch.cat([data_over, data_under], dim=1)

                output=diffusion.Fusion_net.test_Fusion(input, device)
                        
                B,C,H,W=over_image_YUV.shape
                YUV = diffusion.tensor2fu(output[:,:,:H, :W], min_max=(0, 1)).astype(np.uint8)
                img_save(YUV, img_name.split(sep='.')[0], test_out_folder)
            
                        