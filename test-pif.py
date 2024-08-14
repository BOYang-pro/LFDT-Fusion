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
from tqdm import tqdm
from util.img_read_save import img_save
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_pif.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    #=====================================================================
    #           Parsing command
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    for dataset_name in ["PIF"]:
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('./dataset/test/PIF/', dataset_name)
        test_out_folder = os.path.join('./dataset/test_result/PIF/', dataset_name+'/')
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
            
        diffusion = Model.create_model(opt, args.local_rank)

        device = torch.device(f'cuda' if opt['gpu_ids'] is not None else 'cpu')
        diffusion.Fusion_net.to(device)
        diffusion.Fusion_net.eval()
        min_max = (-1, 1)
        with torch.no_grad():
            for img_name in tqdm(os.listdir(os.path.join(test_folder, "DoLP"))):
                S0_image = Image.open(os.path.join(test_folder, "S0", img_name)).convert('RGB')
                DoLP_image = Image.open(os.path.join(test_folder, "DoLP", img_name)).convert('RGB')

                S0_image = (ToTensor()(S0_image) * (min_max[1] - min_max[0]) +min_max[0]).unsqueeze(0).cuda()
                DoLP_image = (ToTensor()(DoLP_image) * (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0).cuda()

                S0_image_YUV = RGB2YCrCb(S0_image)

                data_S0=S0_image_YUV[:, 0:1, :, :]
                data_DoLP=DoLP_image[:, 0:1, :, :]

                input = torch.cat([data_S0, data_DoLP], dim=1)

                output=diffusion.Fusion_net.test_Fusion(input, device)
                B,C,H,W=S0_image_YUV.shape
                YUV= YCrCb2RGB(torch.cat((output[:,:,:H, :W], S0_image_YUV[:, 1:2, :, :], S0_image_YUV[:, 2:3, :, :]), dim=1))
                RGB = diffusion.tensor2fu(YUV, min_max=(0, 1)).astype(np.uint8)
                output=diffusion.tensor2fu(output, min_max=(0, 1)).astype(np.uint8)
                img_save(RGB, img_name.split(sep='.')[0], test_out_folder)