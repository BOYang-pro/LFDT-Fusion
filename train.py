import os

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
import random
from util.visualizer import Visualizer
import numpy as np
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train.json',help='JSON file for configuration')
    parser.add_argument('-local_rank', '--local_rank', type=int)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--sample_selected', type=str, default=None, help='select diffusion sampler')
    parser.add_argument('--model_selected', type=str, default=None, help='select diffusion network')
    parser.add_argument('--batch_size', type=int, default=None, help='set batch_size')
    parser.add_argument('--fusion_task', type=str, default=None, help='set fusion_task')
    parser.add_argument('--strategy', type=str, default=None, help='set fusion strategy')
    #=====================================================================
    #           First：Set parameters
    #======================================================================
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    opt['model']['Fusion']['sample_selected']=args.sample_selected
    opt['model']['Fusion']['model_selected']=args.model_selected
    opt['datasets']['train']['batch_size']=args.batch_size
    opt['model']['fusion_task']=args.fusion_task
    opt['model']['Fusion']['mode']=args.strategy
    visualizer = Visualizer(opt)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # Set up the training logger
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    ##################################################
    #              Initialize distributed operations
    #################################################
    gpus=len(opt['gpu_ids'])
    dist.init_process_group(backend='gloo',world_size=gpus,rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)
    ##################################################
    #             Read dataset
    #################################################
    phase = 'train'
    dataset_opt=opt['datasets']['train']
    batchSize = dataset_opt['batch_size']
    train_set = Data.create_dataset_Fusion(dataset_opt, phase,opt)
    if phase == 'train':
        train_sampler=DistributedSampler(train_set,shuffle=True)  #Random sampling
        train_loader =torch.utils.data.DataLoader(
            train_set,
            batch_size=batchSize,
            shuffle=dataset_opt['use_shuffle'],               
            sampler=train_sampler,
            num_workers=dataset_opt['num_workers'],            #Multiple threads to load data for efficiency
            pin_memory=True,                                   #Load data into GPU memory to speed up training
        )
    training_iters = int(ceil(train_set.length / float(batchSize*gpus)))
    #Set the cosine annealing learning rate
    original_list=opt['train']['scheduler']['periods']
    result_list = [x * training_iters for x in original_list]
    opt['train']['scheduler']['periods']=result_list
    if args.local_rank==0:
        logger.info('Initial Dataset Finished')
    #Random seed
    seed = random.randint(0, 10000)
    if args.local_rank == 0:
        print("current random seed: ", seed)
    torch.cuda.manual_seed_all(seed)
    # Instantiation model
    diffusion = Model.create_model(opt,args.local_rank)
    if args.local_rank == 0:
        logger.info('Initial Model Finished')
   

    ################################################################
    ###                          train                            ###
    ################################################################
    current_step = diffusion.begin_step
    start_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    if opt['path']['resume_state']:
        if args.local_rank == 0:
            print('Resuming training from epoch: {}, iter: {}.'.format(start_epoch, current_step))

    for current_epoch in range (start_epoch,n_epoch):
        

        train_sampler.set_epoch(current_epoch)     #Make the data random
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            diffusion.update_learning_rate(current_epoch*train_set.length+istep*batchSize, warmup_iter=opt['train'].get('warmup_iter', -1))

            #
            #          logging
            if args.local_rank == 0:
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    t = (time.time() - iter_start_time) / batchSize
                    lr_log=diffusion.get_current_learning_rate()
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, lr_log[0], 'Train')
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)

            #=========================
            #                  val
            #=========================

                if (istep+1) % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                    diffusion.test_fusion()
                    visuals = diffusion.get_current_test()
                    visualizer.display_current_results(visuals, current_epoch, True)

        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            if args.local_rank==0:
                diffusion.save_network(current_epoch, current_step)
    dist.destroy_process_group()  