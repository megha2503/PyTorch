#mixup, label smoothing, Cosine scheduling, multistepLR

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
import os
from tqdm import tqdm
import torch.nn.functional as F
from numpy.core.fromnumeric import resize

torch.cuda.empty_cache()
torch.cuda.is_available()
torch.cuda.set_device(2)

data_dir = '/data/megha/meghu/May/FGVC/camera_ready_VGVC/stem'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
test_dir = data_dir + '/test'

batch_size = 32
#learning_rate = 1e-3

transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transforms)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def imshow(inp, title=None):
    
    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
images, labels = next(iter(train_dataloader)) 
print("images-size:", images.shape)

out = torchvision.utils.make_grid(images)
print("out-size:", out.shape)

imshow(out, title=[train_dataset.classes[x] for x in labels])

net = models.resnet18(pretrained=True)
net = net.cuda()
net

conf_mtrx = []
conf_mtrx.append([0])

def confusion_matrix_init():
        conf_mtrx = []
        conf_mtrx.append([0])

        for i in os.listdir(train_dir):
            conf_mtrx[0].append(int(i))

        for i in os.listdir(train_dir):
            conf_mtrx.append([int(i)])

        for i in range(1,1+len(os.listdir(train_dir))):
            for j in range(1,1+len(os.listdir(train_dir))):
                conf_mtrx[i].append(0)
        return conf_mtrx


def confusion_matrix(target_t,pred_t,conf_mtrx):
        
        ind_i,ind_j = -1,-1

        for i in range(len(target_t)):
            
            for j in range(1,len(conf_mtrx[0])):
                if conf_mtrx[0][j] == target_t[i]:
                    ind_j = j
                    break
            for j in range(1,len(conf_mtrx)):
                if conf_mtrx[j][0] == pred_t[i]:
                    ind_i = j
                    break
            
            conf_mtrx[ind_i][ind_j] = 1 + conf_mtrx[ind_i][ind_j]
        
        df = pd.DataFrame(conf_mtrx)
        df.to_csv(os.path.join(data_dir,"mixup_Multistep100_stem_conf_mtrx18.csv"))

#Mixup: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

#LabelSmoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


criterion = LabelSmoothingCrossEntropy(reduction='sum')

#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
'''
Intializing MultiStep/CosineAnnealing LearningRate Scheduler
'''
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,100], gamma=0.1)

def accuracy(out, labels):
    _,pred = torch.max(out, dim=1)
    return torch.sum(pred==labels).item()

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 161)
net.fc = net.fc.cuda() 
#if use_cuda else net.fc


n_epochs = 100
print_every = 20
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
test_loss_list = []
accuracy = []
df_data = []
df_conf_matrix = []
df_data_accurate = []


total_step = len(train_dataloader)
print(total_step)
v = len(valid_dataloader)
v

patience = 0
for epoch in range(1, n_epochs+1):
    conf_mtrx = confusion_matrix_init()
    print(epoch)
    running_loss = 0.0
    correct = 0
    total=0
    print(f'Epoch {epoch}\n')
    net.train()

    for batch_idx, (data_, target_) in tqdm(enumerate(train_dataloader)):

        data_, target_ = data_.to(device), target_.to(device)
        data_, targets_a, targets_b, lam = mixup_data(data_, target_,
                                                       1., use_cuda=True)
        

        outputs = net(data_)
        #loss = criterion(outputs, target_)

        #assign mixup_criterion to the loss, to use the regular loss, uncomment the above line
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        running_loss += loss.item()
        _,predicted = torch.max(outputs, dim=1)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #correct += torch.sum(pred==target_).item()

        total += target_.size(0)
    scheduler.step()
    curr_lr = scheduler.get_last_lr()



    train_acc.append(100 * correct / total)
    train_loss.append(running_loss/total_step)
    print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')

    batch_loss = 0
    total_t=0
    correct_t=0
    with torch.no_grad():
        net.eval()
        for data_t, target_t in tqdm((valid_dataloader)):

            data_t, target_t = data_t.to(device), target_t.to(device)
            outputs_t = net(data_t)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _,pred_t = torch.max(outputs_t, dim=1)
            correct_t += torch.sum(pred_t==target_t).item()
            confusion_matrix(target_t.tolist(),pred_t.tolist(),conf_mtrx)
            total_t += target_t.size(0)

        val_acc.append(100 * correct_t/total_t)
        val_loss.append(batch_loss/(len(valid_dataloader)))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), os.path.join('/data/megha/meghu/May/FGVC/BoT','mixup_MultiStep100_stem.pt'))
            print('Improvement-Detected, save-model')
            df_data_accurate.append([epoch,np.mean(train_loss),(100 * correct/total),np.mean(val_loss),(100 * correct_t/total_t)])
            df_conf_matrix.append([epoch,total,correct,total_t,correct_t])
            



#     df_data.append([np.mean(train_loss),(100 * correct/total),np.mean(val_loss),(100 * correct_t/total_t)])
#     net.train()

#     if epoch >1 and (abs(df_data[epoch-1][0] - df_data[epoch-1][2])) > (abs(df_data[epoch-2][0] - df_data[epoch-2][2])):
#         patience = patience + 1
#     else:
#         patience = 0

#     if patience > 4:
#         print("the model is not improving. ", epoch)
    net.eval()
    test_loss = 0
    correct_tt = 0
    total_tt = 0
    #turn off gradients for validation
    with torch.no_grad():
        for data, target in test_dataloader:
            #forward pass
            data, target = data.to(device), target.to(device)
            output = net(data)
            #validation batch loss
            loss = criterion(output, target)
            test_loss += loss.item()
            #calculated the accuracy
            predicted = torch.argmax(output,1)
            correct_tt += (predicted==target).sum().item()
            total_tt += target.size(0)

    #printing test results
    test_loss /= total_tt
    test_loss_list.append(test_loss)
    accuracy.append(correct_tt/total_tt * 100)
    #     accuracy = correct_tt/len(test_dataloader)
    print(f'test loss:{test_loss}..Test Accuracy: {accuracy} ')
        #break
    df_data.append([np.mean(train_loss),(100 * correct/total),np.mean(val_loss),(100 * correct_t/total_t), np.mean(test_loss), accuracy])


print(test_loss_list)
print(accuracy)


df = pd.DataFrame(df_data,columns=["train_loss","train_acc","val_loss","val_acc","test_loss","accuracy"])
df.to_csv(os.path.join('/data/megha/meghu/May/FGVC/BoT',"eval_mixup_Multistep100_stem.csv"))

df2 = pd.DataFrame(df_conf_matrix,columns=["epoch","train_total","train_correct","val_total","val_correct"])
df2.to_csv(os.path.join(data_dir,"mixup_Multistep100_stem_conf_mtrx18.csv"))

fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation Accuracy:HDL")
plt.plot(train_acc, label='train')
plt.plot(accuracy, label='test')
plt.plot(val_acc, label='validation')

plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
plt.savefig('/data/megha/meghu/May/FGVC/BoT/acc_mixup_Multistep100_stem.png')
#plt.savefig(r'/data/megha/meghu/v4_flower/accuracy_fl161_80ep.png')


fig = plt.figure(figsize=(20,10))
plt.title("Train-Validation loss:stem")
#plt.plot(train_acc, label='train')
plt.plot(train_loss, label='train_loss')
#plt.plot(accuracy, label="test_acc")
plt.plot(test_loss_list, label="test_loss")
plt.plot(val_loss, label='val_loss')
#plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')
plt.savefig(r'/data/megha/meghu/May/FGVC/BoT/loss_mixup_Multistep100_stem.png')