# Copyright (c) Phigent Robotics. All rights reserved.

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import model.networks as networks
from model.base_model import BaseModel
from collections import OrderedDict
from util.util import YCrCb2RGB

class Diffusion_Fusion_Model(BaseModel):
    def __init__(self,opt,local_rank):
        super(Diffusion_Fusion_Model, self).__init__(opt)
        # define Network
        self.Fusion_net = networks.define_Network(opt,local_rank)
        self.local_rank=local_rank
        self.schedule_phase = None
        self.centered = opt['datasets']['centered']

        # set loss and load resume state
        self.set_loss()

        if self.opt['phase'] == 'train':
            self.Fusion_net.train()
            train_opt = self.opt['train']
            
            optim_params = list(self.Fusion_net.parameters())
            # Set the optimizer
            self.optG = torch.optim.AdamW(optim_params, lr=train_opt["optimizer"]["lr"], betas=(0.9, 0.999),weight_decay=train_opt["optimizer"]["weight_decay"])
            self.optimizers.append(self.optG)
            # Set the learning rate
            self.setup_schedulers()
            self.log_dict = OrderedDict()
        self.load_network()
        # self.print_network(self.Fusion_net)

    def feed_data(self, data):
        for key in data:
            data[key] = data[key].cuda(self.local_rank)
        self.data =data
        

    def optimize_parameters(self):
    
        self.optG.zero_grad()
        output = self.Fusion_net(self.data['vis'], self.data['ir'])
        # Fusion result
        self.Fusion_result = YCrCb2RGB(torch.cat((output['Fusion'], self.data['fuse_VU'][:, 0:1, :, :], self.data['fuse_VU'][:, 1:2, :, :]), dim=1))

        loss_gradient = output['loss_gradient']
        loss_l1 = output['loss_l1']
        loss_SSIM = output['loss_SSIM']
        dif_loss = output['dif_loss']
      
        loss = output['loss']
        #Averaging loss
        reduce_loss_gradient=self.reduce_tensor(loss_gradient.data)
        reduce_loss_l1=self.reduce_tensor(loss_l1.data)
        reduce_loss_SSIM=self.reduce_tensor(loss_SSIM.data)
        reduce_dif_loss=self.reduce_tensor(dif_loss.data)
        reduce_loss=self.reduce_tensor(loss.data)

        # Back propagation
        loss.backward()
        self.optG.step()
       
        # Set log
        self.log_dict['l_dif'] = reduce_dif_loss.item()
        self.log_dict['l_ssim'] =reduce_loss_SSIM.item()
        self.log_dict['l_1'] = reduce_loss_l1.item()
        self.log_dict['l_g'] =reduce_loss_gradient.item()
        self.log_dict['l_tot'] = reduce_loss.item()


    def reduce_tensor(self, tensor: torch.Tensor):
        "Average over multiple processes"
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt
    
    def test_fusion(self):
        self.Fusion_net.eval()
       
        input = torch.cat([self.data['vis'], self.data['ir']], dim=1)
     
        if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
            self.output = self.Fusion_net.module.test_Fusion(input, self.device)
            self.output = YCrCb2RGB(torch.cat((self.output, self.data['fuse_VU'][:, 0:1, :, :], self.data['fuse_VU'][:, 1:2, :, :]), dim=1))
        else:
            self.output = self.Fusion_net.test_Fusion(input, self.device)
            self.output = YCrCb2RGB(torch.cat((self.output, self.data['fuse_VU'][:, 0:1, :, :], self.data['fuse_VU'][:, 1:2, :, :]), dim=1))
        self.Fusion_net.train()
        
    

    # 设置损失
    def set_loss(self):
        if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
            self.Fusion_net.module.set_loss(self.device)
        else:
            self.Fusion_net.set_loss(self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['vis'] = self.tensor2im(self.data['vis_rgb'], min_max=(0, 1))
        out_dict['ir'] = self.tensor2im(self.data['ir_rgb'], min_max=(0, 1))
        out_dict['Fusion'] = self.tensor2fu(self.Fusion_result, min_max=(0, 1))

        return out_dict

    def get_current_test(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)

        out_dict['vis'] = self.tensor2im(self.data['vis_rgb'], min_max=(0, 1))
        out_dict['ir'] = self.tensor2im(self.data['ir_rgb'], min_max=(0, 1))
        out_dict['Fusion'] = self.tensor2fu(self.output, min_max=(0, 1))
        return out_dict

    def tensor2im(self, image_tensor, imtype=np.float32, min_max=(-1, 1)):
        # (1, 3, 256, 256)===>(3, 256, 256)
        image_numpy = image_tensor[:1, :, :, :].squeeze(0).detach().clamp_(-1, 1).float().cpu().numpy()
        image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        nc, nh, nw = image_numpy.shape

        if nc == 1:
            tmp = np.zeros((nh, nw, 1))
            tmp = image_numpy.transpose(1, 2, 0)
            tmp = np.tile(tmp, (1, 1, 3))
            image_numpy = tmp
        elif nc == 3:
            tmp = np.zeros((nh, nw, 3))
            tmp = image_numpy.transpose(1, 2, 0)
            image_numpy = tmp

        image_numpy -= np.amin(image_numpy)
        #image_numpy = (image_numpy / np.amax(image_numpy))
        image_numpy = (image_numpy /2.0)

        image_numpy = image_numpy * 255.0
        return image_numpy.astype(imtype)

    def tensor2fu(self, image_tensor, imtype=np.float32, min_max=(-1, 1)):
        # (1, 3, 256, 256)===>(3, 256, 256)
        image_numpy = image_tensor[:1, :, :, :].squeeze(0).detach().clamp_(-1, 1).float().cpu().numpy()
        image_numpy = (image_numpy - min_max[0]) / (min_max[1] - min_max[0])  

        nc, nh, nw = image_numpy.shape

        if nc == 1:
            tmp = np.zeros((nh, nw, 1))
            tmp = image_numpy.transpose(1, 2, 0)
            tmp = np.tile(tmp, (1, 1, 3))
            image_numpy = tmp
        elif nc == 3:
            tmp = np.zeros((nh, nw, 3))
            tmp = image_numpy.transpose(1, 2, 0)
            image_numpy = tmp
        image_numpy -= np.amin(image_numpy)
        #image_numpy = (image_numpy / np.amax(image_numpy))
        image_numpy = (image_numpy /2.0)

        image_numpy = image_numpy * 255.0
        return image_numpy.astype(imtype)

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.Fusion_net
        if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)


    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            print(load_path)
            genG_path = load_path

            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.Fusion_net
            if isinstance(self.Fusion_net, nn.parallel.DistributedDataParallel):
                network = network.module
            network.load_state_dict(torch.load(genG_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']




