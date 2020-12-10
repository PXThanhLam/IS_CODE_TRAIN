from data_loader import ImagesDataset
import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def train(args):
    resnet=models.resnet101(pretrained=True).cuda()
    # checkpoint = torch.load('checkpoint/byol_check_out.pt')
    # resnet.load_state_dict(checkpoint)
    learner = BYOL(
        resnet,
        image_size = 256,
        hidden_layer = 'avgpool'
    )
    opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
    train_data=ImagesDataset(folder=args.train_path,image_size = (256,256))
    train_data_loader=DataLoader(train_data,batch_size=args.batchsize,shuffle=True,num_workers=32)
    val_data=ImagesDataset(folder=args.val_path,image_size = (256,256))
    val_data_loader=DataLoader(val_data,batch_size=args.batchsize,shuffle=True)
    curr_best_val=np.inf
    print('START TRAINING')
    for epoch in range(args.epochs):
        loss_log=AvgrageMeter()
        for i, images in enumerate(train_data_loader):
            loss = learner(images.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            n=images.size(0)
            loss_log.update(loss.cpu().data,n)
            if i % args.echo_batches == args.echo_batches - 1:
                print('TRAIN epoch:%d, mini-batch:%3d, Loss= %.4f' % (epoch + 1, i + 1, loss_log.avg))
            
            if (i+1) % (args.val_per_iter) ==0:
                resnet.eval()
                loss_log_val=AvgrageMeter()
                for i, images in enumerate(val_data_loader):
                    with torch.no_grad():
                        loss = learner(images.cuda())
                    n=images.size(0)
                    loss_log_val.update(loss.cpu().data,n)
                    print('VAL epoch:%d, mini-batch:%3d, Loss= %.4f' % (epoch + 1, i + 1, loss_log_val.avg))
                resnet.train()
                if loss_log_val.avg <=curr_best_val:
                    print('SAVE MODEL !!!!')
                    curr_best_val=loss_log_val.avg
                    torch.save(resnet.state_dict(), args.model_save_dir+'/byol_check_out.pt')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../train_byol_bottle', help='Use gpu or cpu')
    parser.add_argument('--val_path', type=str, default='../val_byol_bottle', help='Use gpu or cpu')
    parser.add_argument('--gpu', type=bool, default=False, help='Use gpu or cpu')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='initial batchsize')
    parser.add_argument('--val_per_iter', type=int, default=250, help='how many train epoch per val')
    parser.add_argument('--echo_batches', type=int, default=5, help='how many train epoch per val')
    parser.add_argument('--epochs', type=int, default=40, help='total training epochs')
    parser.add_argument('--model_save_dir', action='store_true', default='checkpoint', help='Model save dir')
    args = parser.parse_args()

    train(args)