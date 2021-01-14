# -*- coding: utf-8 -*-
"""
This is the main file to train the generator and discriminator jointly for 60000 iterations

Created on Tue Aug  4 20:14:30 2020

@author: frm
"""
import numpy as np
import os,sys
from scipy.io import loadmat
from utils import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from model.MoG_QSM import *                                 

Normdata=loadmat('../NormFactor.mat')
CosTrnMean=Normdata['CosTrnMean']
CosTrnStd=Normdata['CosTrnStd']
       
if __name__=="__main__":    
    patch_shape=[48,48,48]
    batchsize=2
    log_dir="logs/GAN/"
    n_epochs=60
    D_data=loadmat('D.mat')
    D=D_data['D']
    
    g_model=define_generator(D,0,Normdata,input_shape=patch_shape,output_size=patch_shape)
    g_model.load_weights("logs/GAN1014/19_0.3728481.h5")
    
    d_model=define_discriminator(input_shape=patch_shape)
    gan_model=define_gan(D,Normdata,g_model,d_model,input_shape=patch_shape)
    
    lines_data=loadmat('lines.mat')
    lines_array=lines_data['index']
    lines=lines_array.tolist()
    
    datapath='train_data/'
    data = data_read(np.arange(1,31), datapath)
    print('ok')
    Joint_train(g_model,d_model,gan_model,lines,data,log_dir,batchsize=batchsize,patch_shape=patch_shape,n_epochs=n_epochs)
    