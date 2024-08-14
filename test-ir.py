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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_ir.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int, default=0)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    #=====================================================================
    #           Parsing command
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    for dataset_name in ["M3FD"]:
        print("The test result of " + dataset_name + ' :')
        test_folder = os.path.join('./dataset/test/IR/', dataset_name)
        test_out_folder = os.path.join('./dataset/test_result/IR/', dataset_name+'/')
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
        diffusion = Model.create_model(opt, args.local_rank)

        device = torch.device(f'cuda' if opt['gpu_ids'] is not None else 'cpu')
        diffusion.Fusion_net.to(device)
        total = sum([param.nelement() for param in diffusion.Fusion_net.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))
        diffusion.Fusion_net.eval()
        min_max = (-1, 1)
        start_time=time.time()
        with torch.no_grad():
            for img_name in tqdm(os.listdir(os.path.join(test_folder, "ir"))):
                visible_image = Image.open(os.path.join(test_folder, "vi", img_name)).convert('RGB')
                infrared_image = Image.open(os.path.join(test_folder, "ir", img_name)).convert('RGB')

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
                img_save(RGB, img_name.split(sep='.')[0], test_out_folder)
        end_time=time.time()
        print(dataset_name,":",start_time-end_time)
            