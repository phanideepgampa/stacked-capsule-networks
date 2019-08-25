import os
import argparse
import logging
import random
import pickle
from collections import namedtuple
import time
import traceback

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets
import torchvision.transforms
import numpy as np
from tqdm import tqdm


import model

np.set_printoptions(precision=4, suppress=True)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def total_loss(output_scae):
    return 0.

def train(args,device,train,test):

    scae = model.SCAE().to(device)

    optimizer = optim.RMSprop(scae.parameters(), lr=args.lr, momentum=0.9,eps=(1/(10*args.batch_size)**2) )

    for epoch in range(args.epochs):    
        ave_loss = 0
        for batch_idx, (x, target) in enumerate(train):
            optimizer.zero_grad()
            x = Variable(x).to(device)
            out = scae(x)

            loss = total_loss(out)
            
            ave_loss = ave_loss * 0.9 + loss.data[0] * 0.1
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train):
                print ('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx+1, ave_loss))
    return


def main():
    seed_everything()

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-5)

    args = parser.parse_args()

    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = torchvision.datasets.MNIST(root=root, train=False, transform=trans, download=True)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=args.batch_size,
                    shuffle=False)

    train(args,device,train_loader,test_loader)

if __name__=="__main__":
    main()