# -*- coding: utf-8 -*-
"""
This file is to train the generator for 60000 iterations 

Created on Wed Jul 29 21:41:55 2020

@author: frm
"""
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
import numpy as np
import os,sys
from scipy.io import loadmat
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from model.MoG_QSM import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
Normdata=loadmat('../NormFactor.mat')
CosTrnMean=Normdata['CosTrnMean']
CosTrnStd=Normdata['CosTrnStd']

def data_read(data_orders, pathname):
    read_data = []
    for i in data_orders:
        data_temp = loadmat(pathname +str(i) + '.mat')
        read_data.append(data_temp)
    print(len(read_data))
    return read_data
    
        
def generate_arrays_from_file(data,lines,batch_size,input_patch_shape=[48,48,48],output_patch_shape=[48,48,48]):
    n=len(lines)
    i=0
    while 1:
        batch_X=[]
        batch_Y=[]
        batch_mask=[]
        
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            #generate inputs and labels
            temp=lines[i]
            subject=temp[0]
            batch_data=data[subject-1]         

            aug_Xdata=batch_data['inputs']
            aug_Ydata=batch_data['labels']
            aug_mask=batch_data['mask']                                 
           
            Xdata=aug_Xdata[int(temp[1]-input_patch_shape[0]/2):int(temp[1]+input_patch_shape[0]/2),int(temp[2]-input_patch_shape[1]/2):int(temp[2]+input_patch_shape[1]/2),int(temp[3]-input_patch_shape[2]/2):int(temp[3]+input_patch_shape[2]/2)]
            mask=aug_mask[int(temp[1]-input_patch_shape[0]/2):int(temp[1]+input_patch_shape[0]/2),int(temp[2]-input_patch_shape[1]/2):int(temp[2]+input_patch_shape[1]/2),int(temp[3]-input_patch_shape[2]/2):int(temp[3]+input_patch_shape[2]/2)]
            Ydata=aug_Ydata[int(temp[1]-output_patch_shape[0]/2):int(temp[1]+output_patch_shape[0]/2),int(temp[2]-output_patch_shape[1]/2):int(temp[2]+output_patch_shape[1]/2),int(temp[3]-output_patch_shape[2]/2):int(temp[3]+output_patch_shape[2]/2)]
            Xdata=Xdata*mask
            
            Ydata=(Ydata-CosTrnMean)/CosTrnStd           

            batch_X_temp = np.expand_dims(Xdata,axis=3)

            batch_Y_temp = np.expand_dims(Ydata, axis=3)
            batch_mask_temp = np.expand_dims(mask, axis=3)

            batch_X.append(batch_X_temp)
            
            batch_Y.append(batch_Y_temp)
            batch_mask.append(batch_mask_temp)
            
            i=(i+1)%n

        yield [np.array(batch_X), np.array(batch_mask)],np.array(batch_Y)
        
            
    
        
if __name__=="__main__":
    log_dir="logs/20201107/"
    D_data=loadmat('D.mat')
    D=D_data['D']
    model=define_generator(D,0,Normdata,input_shape=[48,48,48],output_size=[48,48,48])
    data = data_read(np.arange(1,31), 'train_data/')
    print('ok')
    lines_data=loadmat('lines.mat')
    lines_array=lines_data['index']
    lines=lines_array.tolist()
    num=len(lines) 
    
    
    checkpoint_period=ModelCheckpoint(log_dir+'ep{epoch:03d}-loss{loss:.3f}.h5',
                                      monitor='val_loss',
                                       save_weights_only=True,
                                       save_best_only=True,
                                       period=1
                                        )
  
    #learning rate
    reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=3,
                                verbose=1) 
    #if need early stop
    early_stopping=EarlyStopping(monitor='val_loss',
                                 min_delta=0,
                                 patience=20,
                                 verbose=1)
    tbCallBack=TensorBoard(log_dir="./logs/20201107/tnsboard",write_graph=True,write_images=True,update_freq='batch')

    #loss
    model.compile(loss='mae',
                        optimizer=Adam(lr=1e-3))
                       
    batch_size=2
    num_train=np.int(0.8*num)
    num_val=np.int(num-num_train)
    np.random.shuffle(lines)
    print('Train on {} samples,val on {} samples,with batch size {}.'.format(num_train,num_val,batch_size))
   
   #start train
    model.fit_generator(generate_arrays_from_file(data,lines[:num_train],batch_size),
                       steps_per_epoch=3000,
                       validation_data=generate_arrays_from_file(data,lines[num_train:],batch_size),
                       validation_steps=100,
                       epochs=20,
                       initial_epoch=0,
                       callbacks=[checkpoint_period,reduce_lr,tbCallBack,early_stopping])
    model.save_weights(log_dir+'last1.h5')
   
    
