import os
import torch
import model as Model
import argparse
import core.logger as Logger
import os
import numpy as np
from util.util  import  RGB2YCrCb,YCrCb2RGB
from torchvision.transforms import ToTensor
from util.img_read_save import img_save
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_nir.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    #=====================================================================
    #           Parsing command
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    for dataset_name in ["NIR"]:
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('./dataset/test/NIR/', dataset_name)
        test_out_folder = os.path.join('./dataset/test_result/NIR/', dataset_name+'/')
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
        
        diffusion = Model.create_model(opt, args.local_rank)

        device = torch.device(f'cuda' if opt['gpu_ids'] is not None else 'cpu')
        diffusion.Fusion_net.to(device)
        diffusion.Fusion_net.eval()
        min_max = (-1, 1)
        with torch.no_grad():
            for img_name in tqdm(os.listdir(os.path.join(test_folder, "NIR"))):
                visible_image = Image.open(os.path.join(test_folder, "RGB", img_name)).convert('RGB')
                infrared_image = Image.open(os.path.join(test_folder, "NIR", img_name)).convert('RGB')
                
                visible_image = (ToTensor()(visible_image) * (min_max[1] - min_max[0]) +min_max[0]).unsqueeze(0).cuda()
                infrared_image = (ToTensor()(infrared_image) * (min_max[1] - min_max[0]) + min_max[0]).unsqueeze(0).cuda()

                visible_image_YUV = RGB2YCrCb(visible_image)

                data_VIS=visible_image_YUV[:, 0:1, :, :]
                data_IR=infrared_image[:, 0:1, :, :]

                input = torch.cat([data_VIS, data_IR], dim=1)

                output=diffusion.Fusion_net.test_Fusion(input, device)
                B,C,H,W=visible_image_YUV.shape
                YUV= YCrCb2RGB(torch.cat((output[:,:,:H, :W], visible_image_YUV[:, 1:2, :, :], visible_image_YUV[:, 2:3, :, :]), dim=1))
                RGB = diffusion.tensor2fu(YUV, min_max=(0, 1)).astype(np.uint8)
                output=diffusion.tensor2fu(output, min_max=(0, 1)).astype(np.uint8)
                img_save(RGB, img_name.split(sep='.')[0], test_out_folder)