#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch import nn, optim
from tqdm import tqdm


# In[2]:


torch.cuda.empty_cache()
torch.cuda.is_available()
torch.cuda.set_device(2)


# In[3]:


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


# In[4]:


data_dir = '/data/megha/meghu/May/FGVC/camera_ready_VGVC/HDL'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/val'
test_dir = data_dir + '/test'


# In[5]:


# trainset = datasets.CIFAR10('/data/megha/meghu/train/', download=True, train=True, transform=transform)
# valset = datasets.CIFAR10('/data/megha/meghu/val/', download=True, train=False, transform=transform)

train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms_train)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms)
valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transforms)


# In[6]:


# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# In[7]:


len_trainset = len(train_dataset)
print('train', len_trainset)
len_valset = len(valid_dataset)
print('valid', len_valset)
len_testset = len(test_dataset)
print('test', len_testset)


# In[8]:


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[9]:


conf_mtrx = []
conf_mtrx.append([0])


# In[10]:


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
        df.to_csv(os.path.join(data_dir,"conf_mtrx18.csv"))


# In[11]:


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

dataiter = iter(train_dataloader)
images, labels = dataiter.next()

print(images.shape)

print(labels.shape)

out = torchvision.utils.make_grid(images)
print("out-size:", out.shape)

imshow(out, title=[train_dataset.classes[x] for x in labels])


# In[12]:


resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
    
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 161)
resnet = resnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.0001)


# In[13]:


def train_and_evaluate(model, train_dataloader, valid_dataloader, criterion, optimizer, len_trainset, len_valset, num_epochs=100):
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in (train_dataloader):
            inputs = inputs.to(device)
            #print(inputs.shape)
            labels = labels.to(device)
            #print(labels.shape)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()  
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len_trainset
        epoch_acc = running_corrects.double() * 100 / len_trainset
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss,
             epoch_acc)) 
         
        model.eval()
        running_loss_val = 0.0 
        running_corrects_val = 0
        for inputs, labels in valid_dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs) 
            loss = criterion(outputs,labels)
            _, preds = torch.max(outputs, 1)
            running_loss_val += loss.item() * inputs.size(0)
            running_corrects_val += torch.sum(preds == labels.data)
      
        epoch_loss_val = running_loss_val / len_valset
        epoch_acc_val = 100 * running_corrects_val.double() / len_valset
      
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
      
        print( 'Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss_val, epoch_acc_val))
      
        print()
        print('Best val Acc: {:4f}'.format(best_acc))
        model.load_state_dict(best_model_wts)
        
        PATH = './KD_HDL.pth'
        torch.save(model.state_dict(), PATH)
        
    return model


# In[14]:


resnet_teacher = train_and_evaluate(resnet,train_dataloader,valid_dataloader,criterion,optimizer,len_trainset,len_valset,80)


# In[15]:


net = models.resnet18(pretrained=True)
net = net.cuda()
net

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 161)
net.fc = net.fc.cuda() 
#if use_cuda else net.fc


# In[16]:


dataiter = iter(train_dataloader)
images, labels = dataiter.next()
out = net(images.cuda())
print(out.shape)
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.0001)
#print(out)


# In[17]:


def loss_kd(outputs, labels, teacher_outputs, temparature, alpha):

    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/temparature, dim=1),F.softmax(teacher_outputs/temparature,dim=1))*(alpha*temparature*temparature)+F.cross_entropy(outputs, labels)*(1.-alpha)
    return KD_loss


# In[18]:


def get_outputs(model, dataloader):
    outputs = []
    for inputs, labels in dataloader:

        inputs_batch, labels_batch = inputs.cuda(), labels.cuda()
        output_batch = model(inputs_batch).data.cpu().numpy()
        outputs.append(output_batch)
    return outputs


# In[19]:


def train_kd(model,teacher_out, optimizer, loss_kd, dataloader, temparature, alpha):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for i,(images, labels) in tqdm(enumerate(dataloader)):
        inputs = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        #print(inputs.shape)
        #print(labels.shape)
        outputs = model(inputs)
        outputs_teacher = torch.from_numpy(teacher_out[i]).to(device)
        loss = loss_kd(outputs,labels,outputs_teacher,temparature, alpha)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
    scheduler.step()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100*running_corrects.double() / len(train_dataset)
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def eval_kd(model,teacher_out, optimizer, loss_kd, dataloader, temparature, alpha):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    for i,(images, labels) in enumerate(dataloader):
        inputs = images.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        outputs_teacher = torch.from_numpy(teacher_out[i]).cuda()
        loss = loss_kd(outputs,labels,outputs_teacher,temparature, alpha)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = 100*running_corrects.double() / len(valid_dataset)
    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    #test the model
    model.eval()
    test_loss = 0
    correct_tt = 0
    total_tt = 0
    test_loss_list = []
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
    #accuracy.append(correct_tt/total_tt * 100)
    accuracy = (correct_tt/total_tt * 100)
    print(f'test loss:{test_loss}..Test Accuracy: {accuracy} ')
    return epoch_acc


def train_and_evaluate_kd(model, teacher_model, optimizer, loss_kd, train_dataloader, valid_dataloader, temparature, alpha, num_epochs=100):
    
    teacher_model.eval()
    best_model_wts = copy.deepcopy(model.state_dict())
    outputs_teacher_train = get_outputs(teacher_model, train_dataloader)
    outputs_teacher_val = get_outputs(teacher_model, valid_dataloader)
    print('Teacher’s outputs are computed now starting the training process-')
    best_acc = 0.0
    
    
    for epoch in range(num_epochs):
        #scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
      
      # Training the student with the soft labes as the outputs 
      #from the teacher and using the loss_kd function
      
        train_kd(model, outputs_teacher_train, optim.Adam(net.parameters()),loss_kd,train_dataloader, temparature, alpha)
     
      # Evaluating the student network
        epoch_acc_val = eval_kd(model, outputs_teacher_val, optim.Adam(net.parameters()), loss_kd, valid_dataloader, temparature, alpha)
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            print('Best val Acc: {:4f}'.format(best_acc))
            model.load_state_dict(best_model_wts)
            
    
    return model


# In[20]:


stud = train_and_evaluate_kd(net,resnet_teacher,optim.Adam(net.parameters()),loss_kd,train_dataloader,valid_dataloader,1,0.5,80)


# In[21]:


stud


# In[22]:


net.eval()
test_loss = 0
correct_tt = 0
total_tt = 0
test_loss_list = []
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
#accuracy.append(correct_tt/total_tt * 100)
accuracy = (correct_tt/total_tt * 100)
print(f'test loss:{test_loss}..Test Accuracy: {accuracy} ')
    #print(f'test loss:{test_loss}..Test Accuracy: {test_accu} ')
        #break


# In[ ]:





# In[ ]:




