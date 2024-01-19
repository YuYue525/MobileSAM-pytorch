import os
import numpy as np
import argparse
import random
import cv2
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter  

from dataset import transform, sa1b_dataset
from mobile_sam.modeling import TinyViT

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="/dataset/vyueyu/sa-1b", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')

    # multi gpu settings
    parser.add_argument("--local_rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=200, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default="/dataset/vyueyu/project/MobileSAM", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=50000, help='save iterations')

    args = parser.parse_args()
    return args

def build_model():
    model = TinyViT(img_size=1024, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )
    
    ## load pretrained TinyViT weights, please download from https://github.com/wkcn/TinyViT?tab=readme-ov-file
    # pretrained_weights = torch.load("path_to_pth")["model"]
    # del pretrained_weights["head.weight"]
    # del pretrained_weights["head.bias"]
    # model.load_state_dict(pretrained_weights, strict=False)
    
    return model

def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()
            
def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (imgs, target_feats, mask_paths) in enumerate(test_loader):
            imgs, target_feats = imgs.cuda(args.local_rank), target_feats.cuda(args.local_rank)
            pred_feats = model.module(imgs)
            test_loss += customized_mseloss(pred_feats, target_feats).item()

    return test_loss / len(test_loader)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
def main(args):

    # multi gpu settings
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    # file folder creating
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
        
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    
    # dataset
    train_dirs = ["sa_" + str(i).zfill(6) for i in range(20)]
    val_dirs = ['sa_000020']
    train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform)
    val_dataset = sa1b_dataset(args.dataset_path, val_dirs, transform, args.eval_nums)
    # training sampler
    train_sampler = DistributedSampler(train_dataset)
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False, num_workers=args.num_workers)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # model
    model = build_model()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    total_iters = 0

    for epoch in range(1, args.epochs + 1):
        # new epoch
        if args.local_rank == 0:
            print("------start epoch {}------".format(epoch))
        train_sampler.set_epoch(epoch)

        # training
        model.train()
        for batch_idx, (imgs, target_feats, mask_paths) in enumerate(train_loader):
            total_iters += 1
            
            imgs, target_feats = imgs.cuda(args.local_rank), target_feats.cuda(args.local_rank)
            optimizer.zero_grad()
            pred_feats = model(imgs)
            loss = customized_mseloss(pred_feats, target_feats)
            loss.backward()
            optimizer.step()
            loss = reduce_mean(loss, dist.get_world_size())
            
            # if is master process
            if args.local_rank == 0:
                # print training info
                if (batch_idx + 1) % args.print_iters == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                        epoch, batch_idx * len(imgs) * dist.get_world_size(), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    writer.add_scalar("mse_loss", loss.item(), total_iters)
                
                # save model
                if total_iters % args.save_iters == 0:
                    save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                    print("save model to {}".format(save_path))
                    torch.save(model.module.state_dict(), save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        dist.barrier()
        scheduler.step()

    # save final model
    if args.local_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        writer.close()

if __name__ == "__main__":
    args = parse_option()
    main(args)
