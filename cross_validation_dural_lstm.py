#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Zongyu Li 
cross validation code for Simamese or other network
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
import warnings
import pickle as pkl
import pandas as pd
import time
from nested_para_dural import find_para




class TimeseriesNet_dural_lstm(nn.Module):
    def __init__(self):
        super(TimeseriesNet_dural_lstm,self).__init__()

       #LSTM(INPUTFEATURES, HIDDEN_SIZE, NUM_LAYERS, BATCH_FIRST=TRUE)
        self.festures=26
        self.seq_len =60
        self.layer_dim =1
        self.lstm1 = nn.LSTM(26,512, dropout=0, num_layers=self.layer_dim,batch_first=True)


        self.lstm2 = nn.LSTM(512, 128, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.lstm3 = nn.LSTM(128, 64, dropout=0, num_layers=self.layer_dim,batch_first=True)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p=0.55)
        self.linear1 = nn.Linear(1920,960)
        self.linear2 = nn.Linear(960,480)

        self.linear6 = nn.Linear(480,16)
        self.linear7 = nn.Linear(16,1)
        
        self.initialize_weights()

    def forward(self,l,l1):
        l=l.transpose(1,2).contiguous()
        l1=l1.transpose(1,2).contiguous()
        
        h0_l = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        c0_l = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm1(l,(h0_l.detach(), c0_l.detach()))
        lstm = F.relu(lstm)
        
        h1_l = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        c1_l = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm2(lstm,(h1_l.detach(), c1_l.detach()))
        lstm = F.relu(lstm)
        
        h2_l = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        c2_l = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        lstm,(hn,cn) = self.lstm3(lstm,(h2_l.detach(), c2_l.detach()))
        lstm = F.relu(lstm)
        lstm = self.flat(lstm)
        
        
        h0_l1 = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        c0_l1 = torch.randn(self.layer_dim,l.size(0),512,device=torch.device("cuda:0")).requires_grad_()
        lstm1,(hn1,cn1) = self.lstm1(l1,(h0_l1.detach(), c0_l1.detach()))
        lstm1 = F.relu(lstm1)
        
        h1_l1 = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        c1_l1 = torch.randn(self.layer_dim,lstm.size(0),128,device=torch.device("cuda:0")).requires_grad_()
        lstm1,(hn1,cn1) = self.lstm2(lstm1,(h1_l1.detach(), c1_l1.detach()))
        lstm1 = F.relu(lstm1)
        
        h2_l1 = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        c2_l1 = torch.randn(self.layer_dim,lstm.size(0),64,device=torch.device("cuda:0")).requires_grad_()
        lstm1,(hn1,cn1) = self.lstm3(lstm1,(h2_l1.detach(), c2_l1.detach()))
        lstm1 = F.relu(lstm1)
     
        lstm1 = self.flat(lstm1)
        
        final= torch.abs(torch.sub(lstm,lstm1))
        
        final = F.relu(self.linear1(final))
    
        final = F.relu(self.linear2(final))
   
        final = F.relu(self.linear6(final))
        final = self.linear7(final)
        
        return final
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias,0)



class TimeseriesPairs_train_v2(Dataset):
    def __init__(self, win_len,stride,x_train,y_train):

        '''
        Parameters
        ----------
        win_len : int
            the length of the sliding window
        stride : int
        
         x_train : 1d array of object shape=[x,]
            the time series data.
        y_train : 1d array error or normal

        Returns
        -------
        None.

        '''
        self.L=[]
        self.L2=[]
        self.y=[]
        init_x = x_train
        init_y = y_train
        
        y_values = [val=='err' for val in init_y]
        
        pairs=[]
        pairs_y=[]  # 0 nor/nor 1 error/nor
        for i in np.arange(0,init_x.shape[0]):
            for j in np.arange(i+1,init_x.shape[0]):
                if y_values[i]==0 and y_values[j]==0 :
                    pairs_y.append(0)
                    pairs.append((i,j))
                elif (y_values[i]==0 and y_values[j]==1) or (y_values[i]==1 and y_values[j]==0):
                    pairs_y.append(1)
                    pairs.append((i,j))
                else: continue
        
        for i, indexs in enumerate(pairs):
            ar1,ar2=indexs
            L_data1 = init_x[ar1]
            time_len1 = L_data1.shape[0]
            start1 = (time_len1-win_len)%stride
            y_val=pairs_y[i]
            cur_data_L=[L_data1[i:i+win_len,:].T for i in \
                      np.arange(start1,time_len1-win_len+stride,stride) ]
            first=[]
            for idx,seq in enumerate(cur_data_L):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    first.append(seq)
                    
            L_data2 = init_x[ar2]
            time_len2 = L_data2.shape[0]
            start2 = (time_len2-win_len)%stride
            cur_data_L2=[L_data2[i:i+win_len,:].T for i in \
                      np.arange(start2,time_len2-win_len+stride,stride) ]
            second=[]
            for idx,seq in enumerate(cur_data_L2):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    second.append(seq)
            
            for a in np.arange(0,len(first)):
                for b in np.arange(0,len(second)):
                    self.L.append(first[a])
                    self.L2.append(second[b])
                    self.y.append(y_val)
            
        
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.L2 = np.array(self.L2, dtype=np.float32)


    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.L2[index],self.y[index]



class TimeseriesPairs_test_v2(Dataset):
    def __init__(self,win_len,stride,x_train,y_train,x_test, y_test):
        '''
        

        Parameters
        ----------
        win_len : int
            the length of the sliding window
        stride : int
        
        x_train : 1d array of object shape=[x,]
            the time series data.
        y_train : 1d array error or normal
            
        x_test : 1d array of object shape=[x,]

        y_test : 1d array error or normal
        
        error_mode: 1x5 one hot encoding arrray for the error modes of the test set

        Returns
        -------
        None.

        '''

        self.L=[]
        self.L2=[]
        self.y=[]
        self.idx=np.empty((0,2), int) # test trial, and window idx

        
        y_train = [val=='err' for val in y_train]
        y_test  =[val=='err' for val in y_test]
        
        pairs=[]
        pairs_y=[]  # 0 nor/nor 1 error/nor
        for i in np.arange(0,x_train.shape[0]):
            for j in np.arange(0,x_test.shape[0]):
                #print(f'y_train{y_train[i]},y_test{y_test[j]}\n')
                if y_train[i]==0 and y_test[j]==0 :
                    pairs_y.append(0)
                    pairs.append((i,j))
                    # pair with all the normal trials
                elif (y_train[i]==0 and y_test[j]==1): #or (y_train[i]==1 and y_test[j]==0):
                    pairs_y.append(1)
                    pairs.append((i,j))
                    
                else: continue
        
        for i, indexs in enumerate(pairs):
            ar1,ar2=indexs
            L_data1 = x_train[ar1]
            time_len1 = L_data1.shape[0]
            start1 = (time_len1-win_len)%stride
            y_val=pairs_y[i]
            cur_data_L=[L_data1[i:i+win_len,:].T for i in \
                      np.arange(start1,time_len1-win_len+stride,stride) ]
            first=[] # train array
            for idx,seq in enumerate(cur_data_L):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    first.append(seq)
                    
            L_data2 = x_test[ar2]
            time_len2 = L_data2.shape[0]
            start2 = (time_len2-win_len)%stride
            cur_data_L2=[L_data2[i:i+win_len,:].T for i in \
                      np.arange(start2,time_len2-win_len+stride,stride) ]
            second=[]
            for idx,seq in enumerate(cur_data_L2):
                count_zero=sum(np.array(seq[0,:])==0)/win_len
                if count_zero<0.4:
                    second.append(seq)
            
            for b in np.arange(0,len(second)):
                for a in np.arange(0,len(first)):
                    self.L.append(first[a])
                    self.L2.append(second[b])
                    self.y.append(y_val)
                    self.idx=np.vstack((self.idx,[ar2,b])) # saving the test trial id, and the window
                    # append the error type for this training example
            
            
        self.y = np.array(self.y, dtype=np.float32)
        self.L = np.array(self.L, dtype=np.float32)
        self.L2 = np.array(self.L2, dtype=np.float32)
        # self.idx denote the classification result of each test sliding window
        self.idx = np.array(self.idx, dtype=np.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,index):
        return self.L[index],self.L2[index],self.y[index],self.idx[index,:]







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
    valid = np.array(cur[:,-1])
    return (init_x,init_y,error_mode,valid)


net_type='double_lstm'

Tasks=["NeedlePassing"] #"Suturing",

for Task in Tasks:
    
    if Task=="Suturing":
        all_g= ["G2", "G3","G4","G6","G8","G9"] #"G1"
        data_dir='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/Suturing_euler_normalized_d_witherrormodes_trialid.mat'
    else: 
        all_g=["G4","G6"] #"G1","G2","G3",
        data_dir='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/Needle_Passing_euler_normalized_d_witherrormodes_trialid.mat'
    F1_mean=np.empty((0,1),dtype=float)
    F1_std=np.empty((0,1),dtype=float)
    F1_mean=np.empty((0,1),dtype=float)
    F1_std=np.empty((0,1),dtype=float)
    F1_v_mean=np.empty((0,1),dtype=float)
    F1_v_std=np.empty((0,1),dtype=float)
    time_run=np.empty((0,1),dtype=float)
    for i, G in enumerate(all_g):
        # if i !=0: continue
        
        state=0
        tic = time.perf_counter()

        
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
        precision_result={}
        recall_result={}
        predicted_result={}
        error_types={}
        expected_result={}
        
        for fold in range(len(unique_ids)):
            # if fold !=1 : continue
            
            # model defintion here
            model=TimeseriesNet_dural_lstm()
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
            model.to(device)
            
            train_ids,test_ids = fold_data[fold]
            xtrain=all_x[train_ids]
            ytrain=all_y[train_ids]
            xtest=all_x[test_ids]
            ytest=all_y[test_ids]
            error_type_test=error_mode[test_ids,:]
            error_mode_train=error_mode[train_ids,:]
            subject_train_ids=ids[train_ids]
            
            cur_para=find_para(G,model,xtrain,ytrain,subject_train_ids,error_mode_train,Task,fold,win_len,stride,net_type)
            config = {
            "l1": 8,
            "l2": 32,
            "lr":cur_para['lr'],
            "batch_size":cur_para['batch_size'],
            'epoch':cur_para['epoch']}
            optimizer= torch.optim.Adam(model.parameters(),lr=config["lr"])
            traindata = TimeseriesPairs_train_v2( win_len,stride,xtrain,ytrain)
            trainloader = DataLoader(dataset=traindata, batch_size=int(config["batch_size"])\
                                      ,shuffle=True,num_workers=0)
        
            w = [sum(np.array(ytrain)!='err')/sum(np.array(ytrain)=='err')]
            class_weight=torch.FloatTensor(w).to(device)
            criterion = nn.BCEWithLogitsLoss(class_weight)
            
            testdata = TimeseriesPairs_test_v2( win_len,stride,xtrain,ytrain,xtest,ytest)
            testloader = DataLoader(dataset=testdata, batch_size=int(config["batch_size"])\
                                      ,shuffle=False,num_workers=0)
            for n in range(config['epoch']):
                model.train()
                
                for i, data in enumerate(trainloader,0):
                    local_batch_L,local_batch_L2,local_y = data
                    local_batch_L,local_batch_L2,local_y = local_batch_L.to(device),local_batch_L2.to(device),\
                    local_y.to(device)
                    if local_batch_L.shape[0]!=config["batch_size"]:continue
                   # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs =torch.squeeze(model(local_batch_L,local_batch_L2))
                    #print(f"input size{local_batch_L.shape}")
                    #print(f"y size{local_y.shape}")
                    loss = criterion(outputs.view(-1),local_y.view(-1))
                    loss.backward()
                    optimizer.step()
                print('epoch={}'.format(n))    
            print('finish train at fold={}'.format(fold))
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
            y_idx=np.empty((0,2), int)  # for evaluating per testing window
            model.eval()
            for i, data in enumerate(testloader,0):
                with torch.no_grad():
                    
                    local_batch_L, local_batch_L2,local_y,idx = data
                    local_batch_L,local_batch_L2,local_y = local_batch_L.to(device),\
                        local_batch_L2.to(device),local_y.to(device)
                    # if local_y.dim() == 0: continue
                    if local_y.view(-1).cpu().numpy().size==0: continue
                    outputs =torch.squeeze(model(local_batch_L,local_batch_L2))
                    
                    outputs_val=outputs>0
                    correct+=(outputs_val.view(-1) == local_y.view(-1)).sum().cpu().numpy()
                    total +=outputs.cpu().numpy().size
                    # tn, fp, fn, tp =confusion_matrix(local_y.view(-1).cpu().numpy(),outputs_val.view(-1).cpu().numpy()).ravel()
                    # TP+=tp
                    # FP+=fp
                    # FN+=fn
                    
                    result.extend(outputs_val.view(-1).cpu().numpy())
                    y_expected.extend(local_y.view(-1).cpu().numpy())
                    y_idx=np.vstack((y_idx,idx))
           
            # calculating the voting result
            result = np.array(result)
            y_expected = np.array(y_expected)
            voting=[] # store the result for each testing window
            y_truth=[]
            errortypes=np.empty((0,5), int)
            for idx_r in np.unique(y_idx,axis=0):
                # get the locations of the corresponding test windows
                trial,window=idx_r
                trial=int(trial)
                mask = np.logical_and(y_idx[:,0]==trial,y_idx[:,1]==window)
                y_result_w=result[mask]
                y_expected_w=y_expected[mask][0]
                vote = sum(y_result_w)/len(y_result_w)
                if vote>0.5:
                    voting.append(1)
                else:
                    voting.append(0)
                y_truth.append(y_expected_w)
                errortypes=np.vstack((errortypes,error_type_test[trial,:]))
                
            try:
                TN_v, FP_v, FN_v, TP_v =confusion_matrix(y_truth,voting).ravel()
                aa=[TN_v, FP_v, FN_v, TP_v]
                TN_v, FP_v, FN_v, TP_v=[0.001 if a==0 else a for a in aa]
                precision_v=TP_v/(TP_v+FP_v)
                recall_v=TP_v/(TP_v+FN_v)
                F1_v=2*precision_v*recall_v/(precision_v+recall_v)  
            except ValueError:
                F1_v=0
            
            try:
                TN,FP,FN,TP=confusion_matrix(y_expected,result).ravel()
                aa=[TN, FP, FN, TP]
                TN, FP, FN, TP=[0.001 if a==0 else a for a in aa]
                precision=TP/(TP+FP)
                recall=TP/(TP+FN)
                accuracy=correct/total
                F1=2*precision*recall/(precision+recall)

            except ValueError:
                F1=0

            F1scores[fold]=[F1,F1_v]
            precision_result[fold]=[precision,precision_v]
            recall_result[fold]=[recall,recall_v]
            predicted_result[fold]=voting
            expected_result[fold]=y_truth
            error_types[fold]=errortypes
            # print('fold={}'.format(fold))
            
                     
        F_val=np.empty((0,1))
        F_v_val=np.empty((0,1))           
        for i in F1scores.keys(): 
            F_val=np.append(F_val,F1scores[i][0])
            F_v_val=np.append(F_v_val,F1scores[i][1])
        
        
        F_mean=np.mean(F_val)
        F_std=np.std(F_val)
        F_v_mean=np.mean(F_v_val)
        F_v_std=np.std(F_v_val)
        # append to the list for making the table
        F1_mean=np.append(F1_mean,F_mean)
        F1_std=np.append(F1_std,F_std)
        F1_v_mean=np.append(F1_v_mean,F_v_mean)
        F1_v_std=np.append(F1_v_std,F_v_std)
        # dump the dictionary into binary file
        AllD=[F1scores, precision_result, recall_result, predicted_result,expected_result,error_types]
        save_folder='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/nested_{}'.format(net_type)
        if not os.path.exists(save_folder):
           os.makedirs(save_folder)
        file_name='/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/nested_{}/{}_result_lstm_nested_{}.p'.format(net_type,G, Task)
        pkl.dump(AllD, open(file_name,"wb"))
        toc = time.perf_counter()
        time_run=np.append(time_run,toc-tic)
        print('done_{}_{}'.format(G,Task))
    

    tb={'F1_mean': F1_mean, 'F1_std':F1_std, 'F1_v_mean':F1_v_mean,'F1_v_std':F1_v_std,'time':time_run}
    df=pd.DataFrame(tb,index=all_g)

    df.to_csv('/home/aurora/Documents/try_ml/Graph_mining_data/Siamese/nested_{}/{}input_F1_new_{}.csv'.format(net_type,net_type,Task))
