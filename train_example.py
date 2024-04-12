import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim import lr_scheduler
import datetime
from datetime import datetime
import numpy as np
import copy
from scipy import stats
import random


from torchvision import datasets, transforms, models
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from loss_exCPR import Prototype, exCPRLoss




parser = argparse.ArgumentParser(description='Evaluate feature metrics on pre-trained torchvision architectures')

parser.add_argument('--train-batch-size', type=int, default=8,
                    help='training batch-size to use')
parser.add_argument('--eval-batch-size', type=int, default=8,
                    help='evaluation batch-size to use')
parser.add_argument('--accumulation-steps', type=int, default=8,
                    help='accumulate gradients this many times before updating parameters')
parser.add_argument('--lr-classifier', type=float, default=0.01,
                    help='learning rate for new classifier head')
parser.add_argument('--lr-extractor', type=float, default=0.01,
                    help='learning rate for finetuning feature extractor')
parser.add_argument('--epochs-max', type=int, default=15,
                    help='max epochs for training classifier or finetuning')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight-decay for training')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for training')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')

parser.add_argument('--verbose', default=0, type=int,
                    help="whether to print to std out")



args = parser.parse_args()


kwargsUser = {}
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string

with open('commandline_args.txt', 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

#use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
torch.cuda.empty_cache()
print ("cuda: ", use_cuda)

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()
        m.training=False

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()


#def reset_learnable(m, ft=0):
#    for param in m.extractor.parameters():
#        param.requires_grad = False
#        m.apply(set_bn_eval)

#    if ft:
#        for param in m.extractor.parameters():
#            param.requires_grad = True
#        m.apply(set_bn_train)

#    for param in m.linear.parameters():
#        param.requires_grad = True


class TransferWrapper(nn.Module):
    def __init__(self, extractor, nftr, nclass):
        super(TransferWrapper, self).__init__()

        self.extractor = extractor
        self.nclass = nclass
        self.nftr = nftr
        self.multi_out = 0

        self.do = nn.Dropout(0.2)
        self.linear = nn.Linear(self.nftr, self.nclass)

    def forward(self, x):

        p = self.extractor(x)

        out = self.linear(self.do(p))

        if (self.multi_out):
            return p, out
        else:
            return out


def evaluate(model, loader):
    correct = 0
    total = 0
    loss = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += F.cross_entropy(outputs, labels).item()

    accuracy = 100 * correct / total
    loss /= len(loader)

    print('Accuracy: %.3f %%' % (accuracy))
    print('Loss: %.5f' % (loss))

    return accuracy, loss



def main():

    #Define dataset transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #224 resnet

    final_size = 224
    
    train_transform = transforms.Compose(
                [transforms.RandomResizedCrop(final_size),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)
                 ])



    gen_transform = transforms.Compose(
                [transforms.Resize(final_size+32),
                 transforms.CenterCrop(final_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)
                 ])


    #Gather data
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
    eval_trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=gen_transform)

    nclass=100
    targs_ds = trainset.targets

    #Get pre-trained model
    weights = models.ResNet18_Weights.DEFAULT
    model_ft = models.resnet18(weights=weights)
    
    num_ftrs = model_ft.fc.in_features
    out_ftrs = nclass
    model_ft.fc = nn.Sequential()
    
    #wrap feature extractor with new classification head.  Allow explicit return of feature vectors.
    cur_model = TransferWrapper(model_ft, num_ftrs, out_ftrs).to(device)


    #Define optimizer
    optimizer = optim.SGD([{'params': cur_model.extractor.parameters()},
                          {'params': cur_model.linear.parameters(), 'lr':args.lr_classifier}], lr=args.lr_extractor, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(cur_model.parameters(), lr=0.00002)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # define a trainloader and testloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
    #eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.eval_batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False)

    #Create Prototype object and pass object reference to exCPRLoss module
    prototypes = Prototype(nclass, num_ftrs)
    criterion = exCPRLoss(prototypes)


    #Prepare training
    cur_model.multi_out = 1
    cur_model.train()
    #reset_learnable(cur_model, ft=args.finetune)


    num_epochs = args.epochs_max
    train_accs = []
    train_losses = []
    test_accs = []
    test_losses = []
    acc_steps = args.accumulation_steps


    cur_model.zero_grad() 

    for epoch in range(1,num_epochs+1):

        running_loss = 0.0
        cur_model.multi_out = 1

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            ftr_vec, outputs = cur_model(inputs)
            loss = criterion(ftr_vec, outputs, labels)


            loss = loss / acc_steps
            loss.backward()

            if (i+1) % acc_steps == 0:
                prototypes.step()             
                optimizer.step()                            
                cur_model.zero_grad()                       

            #if i < steps_per_epoch:
            #    scheduler.step()

            running_loss += loss.item()

            if i % 1000 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(labels), len(trainloader.dataset),
                       100. * i / len(trainloader), loss.item()))

        scheduler.step()
        train_losses.append(running_loss/len(trainloader))

        #print(f'Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

        print(f'Epoch {epoch}/{num_epochs}, Loss: {running_loss/len(trainloader):.3f}')
        #accuracy_train, loss_train = evaluate(cur_model, eval_trainloader)
        cur_model.eval()
        cur_model.multi_out = 0
        acc_train, loss_train = evaluate(cur_model, eval_trainloader)
        acc_test, loss_test = evaluate(cur_model, trainloader)
        cur_model.train()
        
        train_accs.append(acc_train)
        train_losses.append(loss_train)
        test_accs.append(acc_test)
        test_losses.append(loss_test)


if __name__ == '__main__':
    main()

