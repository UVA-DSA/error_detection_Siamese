#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 17:07:05 2021

@author: Zongyu LI
"""

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import scipy.io
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_recall_fscore_support
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pickle as pkl
import pandas as pd
import warnings
from nested_para import find_para



class TimeseriesNet(nn.Module):
    def __init__(self,l1,l2):
        super(TimeseriesNet,self).__init__()
        self.stage_1_conv = nn.Conv1d(26,64,kernel_size=5,stride=1)
        self.stage_1_pool = nn.MaxPool1d(2,2)
        self.stage_1_drop = nn.Dropout(p=0.2)
        self.stage_1_norm = nn.BatchNorm1d(64)
        self.stage_2_conv = nn.Conv1d(64,128,kernel_size=5,stride=1)
        self.stage_2_pool = nn.MaxPool1d(2,2)
        self.stage_2_drop = nn.Dropout(p=0.2)
        self.stage_2_norm = nn.BatchNorm1d(128)

        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(512,l1)
        self.linear2 = nn.Linear(l1,l2)
        self.linear3 = nn.Linear(l2,32)
        self.linear4 = nn.Linear(32,1)

        
        self.initialize_weights()
        
        #barch normalization
        #self.stage_2_conv = nn.Conv1d()
    def forward(self,l):
        l = F.relu(self.stage_1_conv(l))
        l = self.stage_1_pool(l)
        l = self.stage_1_drop(l)
        l = self.stage_1_norm(l)
        l = F.relu(self.stage_2_conv(l))
        l = self.stage_2_pool(l)
        l = self.stage_2_drop(l)
        l = self.stage_2_norm(l)
        l = self.flat(l)
        l = F.relu(self.linear1(l))
        l = F.relu(self.linear2(l))
        l = F.relu(self.linear3(l))
        l = self.linear4(l)
        
        return l
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)
                

class TimeseriesData(Dataset):
    def __init__(self, init_x,init_y,error_mode,win_len=30, stride=1):
        ''' 

        Parameters
        ----------
        init_x : array of objects
            each instance of init_x is a multiD array.
        init_y : array of class
        
        error_mode : an Nx5 matrix indicating error type

        win_len : TYPE, optional
            DESCRIPTION. The default is 1.
        stride : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''

        # use a sliding window to create more data per trial
        self.L=[]

        self.y=[]
        self.err=np.empty((0,5),int)
        for idx,data in enumerate(init_x):
            time_len = data.shape[0]
            start = (time_len-win_len)%stride
            y_val=init_y[idx]
            L_data = data
            cur_data_L=[L_data[i:i+win_len,:].T for i in \
                      np.arange(start,time_len-win_len+stride,stride) ]
            for i,seq in enumerate(cur_data_L):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    self.L.append(seq)
                    self.y.append(y_val)
                    self.err=np.vstack((self.err,error_mode[idx,:]))
                    
        self.y = [val=='err' for val in self.y]
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.err = np.array(self.err, dtype=np.float32)
                
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.y[index],self.err[index,:]


def load_data(gesture,data_dir=None):
    '''
    This function load the data and return the training and testset

    Parameters
    ----------
    gesture : char
    data_dir : char
        DESCRIPTION. The default is './data':.

    Returns
    -------
    trainset and testset.

    '''
    mat = scipy.io.loadmat(data_dir)
    cur = mat[gesture]
    init_x = cur[:,0]
    init_y = cur[:,2]
    error_mode = cur[:,3:-1]
    valid = cur[:,-1]
    return (init_x,init_y,error_mode,valid)




net_type='cnn'
Tasks=["Suturing","NeedlePassing"] #

for Task in Tasks:
    # if Task !='Suturing':continue
    if Task=="Suturing":
        all_g= ["G1","G2","G3","G4","G6","G8","G9"]
        data_dir='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/Suturing_euler_normalized_d_witherrormodes_trialid.mat'
    else: 
        all_g=["G1","G2","G3","G4","G6"]
        data_dir='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/Needle_Passing_euler_normalized_d_witherrormodes_trialid.mat'
    F1_mean=np.empty((0,1),dtype=float)
    F1_std=np.empty((0,1),dtype=float)

    for i, G in enumerate(all_g):
        # if i !=0: continue
        # if G!='G2':continue

        gesture=G
        win_len=30
        stride=20
        
        all_x,all_y,error_mode,ids = load_data(G,data_dir)
        unique_ids=np.unique(ids)
        fold_data={}
        for i,idx in enumerate(unique_ids):
            test_loc=np.where(ids==idx)[0]
            train_loc=np.where(ids!=idx)[0]
            fold_data[i]=[train_loc,test_loc]
        F1scores={}
        precision_results={}
        recall_results={}
        predicted_result={}
        error_types={}
        expected_result={}

        for fold in range(len(unique_ids)):
            # if fold !=1 : continue

            
            
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
           
            train_ids,test_ids = fold_data[fold]
            xtrain=all_x[train_ids]
            ytrain=all_y[train_ids]
            xtest=all_x[test_ids]
            ytest=all_y[test_ids]
            error_type_train=error_mode[train_ids,:]
            error_type_test=error_mode[test_ids,:]
            traindata = TimeseriesData(xtrain,ytrain,error_mode[train_ids,:] , win_len=win_len, stride=stride)
            
            
            subject_train_ids=ids[train_ids]
            model = TimeseriesNet(l1=256, l2=16)
            cur_para = find_para(G,model,xtrain,ytrain,subject_train_ids,error_type_train,Task,fold,win_len,stride,net_type)
            model.to(device)
            
            config = {"l1": 256,
        "l2": 16,"lr":cur_para['lr'],
    "batch_size": cur_para['batch_size'],
    "epoch":cur_para['epoch']}
            optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
            trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                                      ,shuffle=True,num_workers=0)
            w = [sum(np.array(ytrain)!='err')/sum(np.array(ytrain)=='err')]
            class_weight=torch.FloatTensor(w).to(device)
            criterion = nn.BCEWithLogitsLoss(class_weight)
            testdata = TimeseriesData(xtest,ytest,error_type_test,win_len=win_len, stride=stride)
            testloader = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                          ,shuffle=False,num_workers=8)
            
            for n in range(config['epoch']):
                model.train()
            
                for i, data in enumerate(trainloader,0):
                    local_batch_L,local_y,er = data
                    local_batch_L,local_y = local_batch_L.to(device),\
                    local_y.to(device)
                    if local_batch_L.shape[0]!=config["batch_size"]:continue
                   # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs =torch.squeeze(model(local_batch_L))
                    #print(f"input size{local_batch_L.shape}")
                    #print(f"y size{local_y.shape}")
                    loss = criterion(outputs.view(-1),local_y.view(-1))
                    loss.backward()
                    optimizer.step()
                    
                    # evaluate this fold's perforamnce in terms of F1 score
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            total = 0
            TP=0
            FP=0
            FN=0
            result=[]
            y_expected=[]
            y_error_type=np.empty((0,5), int)  # for evaluating per testing window
            model.eval()
            for i, data in enumerate(testloader,0):
                with torch.no_grad():
                    
                    local_batch_L, local_y,error_type = data
                    local_batch_L, local_y = local_batch_L.to(device),\
                       local_y.to(device) 
                    # if local_y.dim() == 0: continue
                    if local_y.view(-1).cpu().numpy().size==0: continue
                    outputs =torch.squeeze(model(local_batch_L))
                    
                    outputs_val=outputs>0
                    correct+=(outputs_val.view(-1) == local_y.view(-1)).sum().cpu().numpy()
                    total +=outputs.cpu().numpy().size
                    # tn, fp, fn, tp =confusion_matrix(local_y.view(-1).cpu().numpy(),outputs_val.view(-1).cpu().numpy()).ravel()
                    # TP+=tp
                    # FP+=fp
                    # FN+=fn
                    
                    result.extend(outputs_val.view(-1).cpu().numpy())
                    y_expected.extend(local_y.view(-1).cpu().numpy())
                    y_error_type=np.vstack((y_error_type,error_type))

            try:
                TN, FP, FN, TP =confusion_matrix(y_expected,result).ravel()
                aa=[TN, FP, FN, TP]
                TN, FP, FN, TP=[0.001 if a==0 else a for a in aa]
                precision=TP/(TP+FP)
                recall=TP/(TP+FN)
                accuracy=correct/total
                F1=2*precision*recall/(precision+recall)
            except ValueError:
                F1=0


            F1scores[fold]=F1
            predicted_result[fold]=result
            precision_results[fold]=precision
            recall_results[fold]=recall
            expected_result[fold]=y_expected
            error_types[fold]=y_error_type
    
        F_val=np.empty((0,1))
        R_val=np.empty((0,1))
        P_val=np.empty((0,1))
        
        for i in F1scores.keys(): 
            F_val=np.append(F_val,F1scores[i])
            R_val=np.append(F_val,recall_results[i])
            P_val=np.append(F_val,precision_results[i])
        
        F_mean=np.mean(F_val)
        F_std=np.std(F_val)
        # append to the list for making the table
        F1_mean=np.append(F1_mean,F_mean)
        F1_std=np.append(F1_std,F_std)
        # dump the dictionary into binary file
        AllD=[F1scores, precision_results, recall_results, predicted_result,expected_result,error_types]
        save_folder='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/nested_{}/'.format(net_type)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
 
        file_name='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/nested_{}/{}_result_{}_nested_{}.p'.format(net_type,G,net_type, Task)
        pkl.dump(AllD, open(file_name,"wb"))
        print('Done_{}_{}'.format(G,Task))
    
    
    tb={'F1_mean': F1_mean, 'F1_std':F1_std}
    df=pd.DataFrame(tb,index=all_g)
    df.to_csv('/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/nested_{}/{}input_F1_new_{}.csv'.format(net_type,net_type,Task))
               
