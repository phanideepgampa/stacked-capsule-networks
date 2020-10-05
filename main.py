import os
import argparse
import logging
import random
import pickle
from collections import namedtuple
import time
import traceback

import torch
import torch.nn as nn
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


def evaluate(model,train,test,K,device):
    model.eval()
    with torch.no_grad():
        prev_max = -1e+6*torch.ones(K).to(device)
        prev_labels = -1*torch.LongTensor(K).fill_(1).to(device)
        for batch_idx, (x, target) in tqdm(enumerate(train),desc="train"):
            x = Variable(x).to(device)
            target = Variable(target).to(device)
            out = model(x,device,mode='test')
            a_k = out[2].squeeze(-1) #(B,K)
            max_act,max_ex = a_k.max(0).values.view(-1),a_k.max(0).indices.view(-1)  #(K)
            if (max_act>prev_max).sum()!=0:
                for i in range(0,K):
                    if max_act[i]>prev_max[i]:
                        prev_labels[i]=target[max_ex[i]]
                        prev_max[i]=max_act[i]


    count = 0 
    total_count = 0
    with torch.no_grad():
        for batch_idx, (x, target) in tqdm(enumerate(test),desc='test'):
            x = Variable(x).to(device)
            target = Variable(target).to(device)
            out = model(x,device,mode='test')
            a_k = out[2].squeeze(-1) #(B,K)
            max_act,max_ex = a_k.max(-1).values.view(-1),a_k.max(-1).indices.view(-1)
            pred = prev_labels[max_ex]
            count+=(pred == target.data).sum()
            total_count += x.data.size()[0]
    accuracy = count.item()/total_count
    return accuracy


def train(args,train,test,device):

    scae = model.SCAE()
    if torch.cuda.device_count() > 1:
        scae = nn.DataParallel(scae)
    scae.to(device)
    total_loss = model.SCAE_LOSS()
    model_name = "model/scae.model"
    log_name = "log/SCAE"
    prev_best_accuracy = 0.
    K = args.K
    C = args.C
    B = args.batch_size
    if args.load_ext:
        model_name = args.model_file
        print("loading existing model:->%s" % model_name)
        scae = torch.load(model_name, map_location=lambda storage, loc: storage)
        scae.to(device)     
        log_name = 'log/'+model_name.split('/')[-1]
        prev_best_accuracy = evaluate(scae,train,test,K=K,device=device)
       
    logging.basicConfig(filename='%s.log' % log_name,
                        level=logging.DEBUG, format='%(asctime)s %(levelname)-10s %(message)s')

    optimizer = optim.RMSprop(scae.parameters(), lr=args.lr, momentum=0.9,eps=(1/(10*args.batch_size)**2) )
    k_c = torch.tensor(float(K/C)).to(device)
    b_c = torch.tensor(float(B/C)).to(device)
    for epoch in range(args.epochs):    
        ave_loss = 0
        for batch_idx, (x, target) in tqdm(enumerate(train)):
            optimizer.zero_grad()
            scae.train()
            x = Variable(x).to(device)
            out = scae(x,device,mode='train')

            loss = total_loss(out,b_c=b_c,k_c=k_c)
            
            ave_loss = ave_loss * 0.9 + loss.mean().data * 0.1
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(scae.parameters(),5)
            #print(loss)
            optimizer.step()
            if (batch_idx+1) % 50 == 0 or (batch_idx+1) == len(train):
                logging.info('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(
                    epoch, batch_idx+1, ave_loss))

        if (epoch+1)%50 == 0:
            scae.eval()
            accuracy = evaluate(scae,train=train,test=test,K=K,device=device)
            if accuracy>prev_best_accuracy:
                prev_best_accuracy = accuracy
                torch.save(scae, model_name)
                logging.debug("saving model"+str(model_name)+" "+"with test_accuracy:"+ str(accuracy))
            logging.debug('epoch ' + str(epoch) + 'test-accuracy: '
                        + str(accuracy))
    return


def main():
    seed_everything()

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_file', type=str, default='model/scae.model')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--K', type=int, default=24)
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--load_ext', action = "store_true")

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

    if args.mode == 'train':
        train(args,train=train_loader,test=test_loader,device=device)
    else:
        model_name = args.model_file
        print("loading existing model:->%s" % model_name)
        scae = torch.load(model_name, map_location=lambda storage, loc: storage)
        scae.to(device) 
        accuracy = evaluate(scae,train=train_loader,test=test_loader,K=args.K,device=device)
        print("accuracy: %0.4f"%accuracy)


if __name__ == "__main__":
    main()
