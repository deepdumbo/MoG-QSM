# -*- coding: utf-8 -*-
"""
This file contains some supporting functions used in network training.

Created on Thu Jul 30 20:34:07 2020

@author: frm
"""

import numpy as np
import os,sys
from scipy.io import loadmat
import nibabel as nib
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from test.test_tools import *
from model.MoG_QSM import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"

Normdata=loadmat('../NormFactor.mat')
CosTrnMean=Normdata['CosTrnMean']
CosTrnStd=Normdata['CosTrnStd']  

def generate_real_samples(data,lines,batchsize,patch_shape,index): 
    num=len(lines)
    index=index % num
    batch_X=[]          
    for i in range(batchsize):
        temp=lines[i+index]
        subject=temp[0]
        Xdata=data[subject-1]['labels'][int(temp[1]-patch_shape[0]/2):int(temp[1]+patch_shape[0]/2),int(temp[2]-patch_shape[1]/2):int(temp[2]+patch_shape[1]/2),int(temp[3]-patch_shape[2]/2):int(temp[3]+patch_shape[2]/2)]
        Xdata=(Xdata-CosTrnMean)/CosTrnStd        
        batch_X_temp = np.expand_dims(Xdata,axis=-1)
        batch_X.append(batch_X_temp)
    x=np.array(batch_X)
    y=np.ones((batchsize,1))
    return x,y
    
def data_aug(batch_input,input_patch_shape):
    
    if np.random.rand(1)<0.2:
        SNR=np.random.choice([5,10,20,40], 1, p=[0.25, 0.25, 0.25, 0.25])
        Power=np.sum(batch_input**2)/(input_patch_shape[0]**3)
        Noise=np.sqrt(Power/SNR)*np.random.randn(input_patch_shape[0],input_patch_shape[1],input_patch_shape[2])
        batch_input=batch_input+Noise
    return batch_input


def generate_latent_points(data,lines,batchsize,patch_shape,index):
    num=len(lines)
    index=index % num
    batch_X=[]
    batch_mask=[]
    batch_Y=[]
    for i in range(batchsize):
        temp=lines[i+index]
        subject=temp[0]
        Xdata=data[subject-1]['inputs'][int(temp[1]-patch_shape[0]/2):int(temp[1]+patch_shape[0]/2),int(temp[2]-patch_shape[1]/2):int(temp[2]+patch_shape[1]/2),int(temp[3]-patch_shape[2]/2):int(temp[3]+patch_shape[2]/2)]
        mask=data[subject-1]['mask'][int(temp[1]-patch_shape[0]/2):int(temp[1]+patch_shape[0]/2),int(temp[2]-patch_shape[1]/2):int(temp[2]+patch_shape[1]/2),int(temp[3]-patch_shape[2]/2):int(temp[3]+patch_shape[2]/2)]
        Ydata=data[subject-1]['labels'][int(temp[1]-patch_shape[0]/2):int(temp[1]+patch_shape[0]/2),int(temp[2]-patch_shape[1]/2):int(temp[2]+patch_shape[1]/2),int(temp[3]-patch_shape[2]/2):int(temp[3]+patch_shape[2]/2)]
        Ydata=(Ydata-CosTrnMean)/CosTrnStd
        Xdata=data_aug(Xdata,patch_shape)
        Xdata=Xdata*mask
        batch_X_temp = np.expand_dims(Xdata,axis=-1)
        batch_mask_temp = np.expand_dims(mask,axis=-1)
        batch_Y_temp = np.expand_dims(Ydata,axis=-1)
        
        batch_X.append(batch_X_temp)
        batch_mask.append(batch_mask_temp)
        batch_Y.append(batch_Y_temp)
    x1=np.array(batch_X)
    x2=np.array(batch_mask)
    y=np.array(batch_Y)
    return [x1,x2],y
        
def generate_fake_samples(generator,data,lines,batchsize,patch_shape,index):
    x_input,_= generate_latent_points(data,lines,batchsize,patch_shape,index)
    # predict outputs
    X=generator.predict(x_input)
    # create class labels
    y=np.zeros((batchsize,1))
    
    return X,y
    
def compute_rmse(pred,true):
    rmse=100*np.linalg.norm(pred[:]-true[:])/np.linalg.norm(true[:])
    return rmse


def net_valid(generator):
    """This function is called to perform validation during GAN training"""
    data_path1='valid/prisma/'
    input_data = np.array(nib.load(data_path1+'phi1.nii').get_fdata())
    mask = np.array(nib.load(data_path1+'mask1.nii').get_fdata())
    labels = np.array(nib.load(data_path1+'label1.nii').get_fdata())
    [Y_data,output_3Dpatch]=data_predict(generator,input_data,mask, [48,48,48], [48,48,48])
    Y_data=Y_data*CosTrnStd+CosTrnMean
    rmse=compute_rmse(Y_data*mask,labels*mask)
    return rmse
       
def data_read(data_orders, pathname):
    read_data = []
    for i in data_orders:
        data_temp = loadmat(pathname +str(i) + '.mat')
        read_data.append(data_temp)
    print(len(read_data))
    return read_data    

def Dis_train(g_model,d_model,data,lines,batchsize,patch_shape,log_dir,n_epochs=30): 
    """ This function is called to train the discriminator"""
    np.random.shuffle(lines)
    iter_per_epo=1000
    n_steps=iter_per_epo*n_epochs
    num_epo=0
    index=-batchsize/2
    print(index)
    for i in range(n_steps):
        #prepare real and fake samples
        index=int(index+2)
        X_real,y_real=generate_real_samples(data,lines,int(batchsize/2),patch_shape,index) 
        X_fake,y_fake=generate_fake_samples(g_model,data,lines,int(batchsize/2),patch_shape,index)         
        Xdata=np.concatenate((X_real,X_fake),axis=0)
        Ydata=np.concatenate((y_real,y_fake),axis=0)
        # updata discriminator model
        d_loss=d_model.train_on_batch(Xdata,Ydata)
        # summarize loss on this batch
        print('>%d, d1=%.3f' % (i+1, d_loss))
        # record history
        if (i+1) % (iter_per_epo)==0:
            index=-batchsize/2
            np.random.shuffle(lines)
            num_epo=num_epo+1
            d_model.save(log_dir+str(num_epo)+'_'+str(d_loss)+'.h5')
                        
            

def Joint_train(g_model,d_model,gan_model,lines,data,log_dir,batchsize=2,patch_shape=[48,48,48],n_epochs=50):
    '''This function is called to train the generator and discriminator jointly'''    
    #calculate the number of batches per training epoch
    np.random.shuffle(lines)
    iter_per_epo=1000
    n_steps=iter_per_epo*n_epochs
    num_epo=0
    index=-batchsize
    
    for i in range(n_steps):
        #prepare real and fake samples
        for sub_i in range(3):
            index=index+batchsize                  
            X_real,y_real=generate_real_samples(data,lines,batchsize,patch_shape,index) 
            X_fake,y_fake=generate_fake_samples(g_model,data,lines,batchsize,patch_shape,index)  
            # updata discriminator model
            d_loss1=d_model.train_on_batch(X_real,y_real)
            d_loss2=d_model.train_on_batch(X_fake,y_fake)                                       
        
        # update the generator via the discriminator's error
        z_input,y=generate_latent_points(data,lines,batchsize,patch_shape,index)
        z_input.append(y)
        g_loss=gan_model.train_on_batch(z_input,np.ones((batchsize,1)))
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))
        # record history
        if (i+1) % (iter_per_epo)==0:
            index=-batchsize
            np.random.shuffle(lines)
            num_epo=num_epo+1
            L1,L2=net_valid(g_model)
            g_model.save_weights(log_dir+str(num_epo)+'_'+str(L1)+'_'+str(L2)+'.h5')
            d_model.save_weights(log_dir+'DIS/'+str(num_epo)+'.h5')
