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


def evaluate(model,train,test,K,device,num_classes=10):
    """Unsupervised classification accuracy: every object capsule is assigned the label that
    occurs most often among the training examples for which it is the most active capsule;
    a test example is then predicted with the label of its most active capsule."""
    model.eval()
    votes = torch.zeros(K,num_classes,dtype=torch.long,device=device)
    with torch.no_grad():
        for batch_idx, (x, target) in tqdm(enumerate(train),desc="train"):
            x, target = x.to(device), target.to(device)
            caps_presence = model(x,mode='test')[2] #(B,K)
            winner = caps_presence.argmax(-1)
            votes.index_put_((winner,target),torch.ones_like(target),accumulate=True)
    capsule_labels = votes.argmax(-1) #(K)

    count = 0
    total_count = 0
    with torch.no_grad():
        for batch_idx, (x, target) in tqdm(enumerate(test),desc='test'):
            x, target = x.to(device), target.to(device)
            caps_presence = model(x,mode='test')[2] #(B,K)
            pred = capsule_labels[caps_presence.argmax(-1)]
            count += (pred == target).sum().item()
            total_count += x.size(0)
    model.train()
    return count/total_count


def train(args,train,test,device):

    scae = model.SCAE(num_parts=args.M,num_objects=args.K)
    model_name = "model/scae.model"
    log_name = "log/SCAE"
    prev_best_accuracy = 0.
    K = args.K
    if args.load_ext:
        model_name = args.model_file
        print("loading existing model:->%s" % model_name)
        scae.load_state_dict(torch.load(model_name, map_location='cpu'))
        log_name = 'log/'+model_name.split('/')[-1]
    scae.to(device)
    if args.load_ext:
        prev_best_accuracy = evaluate(scae,train,test,K=K,device=device,num_classes=args.C)
    for d in [os.path.dirname(model_name), os.path.dirname(log_name)]:
        if d:
            os.makedirs(d, exist_ok=True)

    logging.basicConfig(filename='%s.log' % log_name,
                        level=logging.DEBUG, format='%(asctime)s %(levelname)-10s %(message)s')

    # TF RMSProp adds epsilon inside the square root, PyTorch outside of it
    eps = (1e-2/args.batch_size**2)**0.5
    optimizer = optim.RMSprop(scae.parameters(), lr=args.lr, momentum=0.9, eps=eps)
    total_loss = model.SCAE_LOSS(num_classes=args.C)
    for epoch in range(args.epochs):
        ave_loss = 0
        scae.train()
        for batch_idx, (x, target) in tqdm(enumerate(train)):
            optimizer.zero_grad()
            x = x.to(device)
            out = scae(x,mode='train')

            loss = total_loss(out)

            ave_loss = ave_loss * 0.9 + loss.item() * 0.1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scae.parameters(),5)
            optimizer.step()
            if (batch_idx+1) % 50 == 0 or (batch_idx+1) == len(train):
                logging.info('==>>> epoch: {}, batch index: {}, train loss: {:.6f}, image ll: {:.2f}, part ll: {:.2f}'.format(
                    epoch, batch_idx+1, ave_loss, out[0].item(), out[1].item()))

        if (epoch+1)%args.eval_every == 0:
            accuracy = evaluate(scae,train=train,test=test,K=K,device=device,num_classes=args.C)
            if accuracy>prev_best_accuracy:
                prev_best_accuracy = accuracy
                torch.save(scae.state_dict(), model_name)
                logging.debug("saving model"+str(model_name)+" "+"with test_accuracy:"+ str(accuracy))
            logging.debug('epoch ' + str(epoch) + ' test-accuracy: '
                        + str(accuracy))
    return


def main():
    seed_everything()

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_file', type=str, default='model/scae.model')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--M', type=int, default=24, help='number of part capsules')
    parser.add_argument('--K', type=int, default=24, help='number of object capsules')
    parser.add_argument('--C', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--load_ext', action = "store_true")

    args = parser.parse_args()

    root = './data'
    if not os.path.exists(root):
        os.mkdir(root)
    # pixels must stay in [0, 1]: they are modelled as a mixture of templates with values in [0, 1]
    trans = torchvision.transforms.ToTensor()
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
        scae = model.SCAE(num_parts=args.M,num_objects=args.K)
        scae.load_state_dict(torch.load(model_name, map_location='cpu'))
        scae.to(device)
        accuracy = evaluate(scae,train=train_loader,test=test_loader,K=args.K,device=device,num_classes=args.C)
        print("accuracy: %0.4f"%accuracy)


if __name__ == "__main__":
    main()
