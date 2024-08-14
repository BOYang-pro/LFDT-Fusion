import torchvision.transforms
import glob
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor
import torchvision.transforms.functional as TF
import os
from PIL import Image
import torch
from util.util  import  randrot, randfilp,RGB2YCrCb

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),  
])


# Read the data set name of the file
def prepare_data_path(dataset_path):
    
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    # Filter the list of all files and folders with the name "bmp "," tif","jpg","png"
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tiff")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    # Sort the list of files and file names
    data.sort()
    filenames.sort()
    return data, filenames

class FusionDataset(Dataset):
    def __init__(self,
                 split,             
                 crop_size=256,    
                 min_max=(-1, 1),
                 ir_path='./PathToIr/',
                 vi_path='./PathToVis/',
                 is_crop=True):  
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.is_crop = is_crop
        self.crop_size = crop_size 
        self.crop = torchvision.transforms.RandomCrop(self.crop_size)
        self.min_max = min_max


        if split == 'train':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))


    def __getitem__(self, index):
        if self.split == 'train':
            visible_image = Image.open(self.filepath_vis[index]).convert('RGB')
            infrared_image = Image.open(self.filepath_ir[index]).convert('RGB')

            visible_image = (ToTensor()(visible_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
            infrared_image = (ToTensor()(infrared_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
            
            
            # (1, C, H, W)+(1, C, H, W)=(2, C, H, W)
            vis_ir = torch.cat([visible_image, infrared_image], dim=1)


            vis_ir = randfilp(vis_ir)
            vis_ir = randrot(vis_ir)
            patch = self.crop(vis_ir)         
            if patch.shape[-1] <= self.crop_size or patch.shape[-2] <= self.crop_size:
                patch = TF.resize(patch, self.crop_size)
            visible_image, infrared_image= torch.split (patch, [3, 3], dim=1)

            visible_image_YUV=RGB2YCrCb(visible_image)
            infrared_image_YUV=RGB2YCrCb(infrared_image)
            return {'vis': visible_image_YUV[:,0:1,:,:].squeeze(0),
                    'fuse_VU': visible_image_YUV[:,1:3,:,:].squeeze(0),
                    'vis_rgb': visible_image.squeeze(0),
                    'ir': infrared_image_YUV[:, 0:1, :, :].squeeze(0),
                    'ir_rgb': infrared_image.squeeze(0),
                    'Index': index}

       


    def __len__(self):
        return self.length
        
        

class FusionDataset_Digtal(Dataset):
    def __init__(self,
                 split,             
                 crop_size=256,    # Resolution used for training
                 min_max=(-1, 1),
                 img1_path=None,
                 img2_path=None,
                 is_crop=True):  
        super(FusionDataset_Digtal, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.is_crop = is_crop
        self.crop_size = crop_size
        # Create a random cropping function
        self.crop = torchvision.transforms.RandomCrop(self.crop_size)
        # To standardize the scope of the image
        self.min_max = min_max

        if split == 'train':
            data_dir_img1 = img1_path
            data_dir_img2 = img2_path
            self.filepath_img1, self.filenames_img1 = prepare_data_path(data_dir_img1)
            self.filepath_img2, self.filenames_img2 = prepare_data_path(data_dir_img2)
            self.split = split
            self.length = min(len(self.filenames_img1), len(self.filenames_img2))


    def __getitem__(self, index):
        if self.split == 'train':
            img1_image = Image.open(self.filepath_img1[index]).convert('RGB')
            img2_image = Image.open(self.filepath_img2[index]).convert('RGB')

            # Convert image to PyTorch tensor and scale it to specified size [-1, 1]
            img1_image = (ToTensor()(img1_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
            img2_image = (ToTensor()(img2_image) * (self.min_max[1] - self.min_max[0]) + self.min_max[0]).unsqueeze(0)
           
            # (1, C, H, W)+(1, C, H, W)=(2, C, H, W)
            img1_img2 = torch.cat([img1_image, img2_image], dim=1)
    
            
            img1_img2 = randfilp(img1_img2)
            img1_img2 = randrot(img1_img2)
            patch = self.crop(img1_img2)  
            if patch.shape[-1] <= self.crop_size or patch.shape[-2] <= self.crop_size:
                patch = TF.resize(patch, self.crop_size)
        
            img1_image, img2_image = torch.split(patch, [3, 3], dim=1)
            
            img1_YUV = RGB2YCrCb(img1_image)
            img2_YUV = RGB2YCrCb(img2_image)
   
 
   
            return {'vis': img1_YUV[:, 0:1, :, :].float().squeeze(0),
                    'fuse_VU': img1_YUV[:, 1:3, :, :].float().squeeze(0),
                    'vis_rgb': img1_image.float().squeeze(0),
                    'ir': img2_YUV[:, 0:1, :, :].float().squeeze(0),
                    'ir_rgb': img2_image.float().squeeze(0),
                    'Index': index}

            
    def __len__(self):
        return self.length
    


