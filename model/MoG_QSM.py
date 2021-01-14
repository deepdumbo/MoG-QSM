# -*- coding: utf-8 -*-
"""
This code will create the generative and  adversarial neural network described in our following paper
MoG-QSM: Model-based Generative Adversarial Deep Learning Network for Quantitative Susceptibility Mapping

Created on Mon Jul 20 14:35:52 2020

@author: frm
"""

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,BatchNormalization,Dropout,Lambda,Add,LeakyReLU,Multiply,Activation,Flatten,Cropping3D,Layer
from keras.layers import Conv3D
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf
import numpy as np

weight_decay=0.0005
def initial_conv(x,filters,kernel_size,strides=(1,1,1),padding='same'):
    """
    This function creates a convolution layer followed by ReLU
    """
    x=Conv3D(filters,kernel_size,strides=strides,padding=padding,
             W_regularizer=l2(weight_decay),
             use_bias=False,
             kernel_initializer='he_uniform')(x)
    
    x=Activation('relu')(x)
    return x


def Res_block(x, k=1, dropout=0.0):
    """
    This function creates a ResBlock
    """
    
    init=x
    
    x = Conv3D(16*k,(3,3,3),padding='same',
               W_regularizer=l2(weight_decay),
               use_bias=False,
               kernel_initializer='he_uniform')(x)
       
    x = BatchNormalization(axis=4)(x)
    x=Activation('relu')(x)    
    
    if dropout > 0.0: x=Dropout(dropout)(x)
    
    x = Conv3D(16*k,(3,3,3),padding='same',
               W_regularizer=l2(weight_decay),
               use_bias=False,
               kernel_initializer='he_uniform')(x)
    x = BatchNormalization(axis=4)(x)
    m = Add()([init,x])
    m=Activation('relu')(m)
    
    return m

def pad_tensor(x,ind1,ind2,ind3,matrix_size):
    """
    This function will pad the output patches to match the size of dipole kernel
    """
    paddings = tf.constant([[0,0],[int((matrix_size[0]-ind1)/2), int((matrix_size[0]-ind1)/2)], [int((matrix_size[1]-ind2)/2), int((matrix_size[1]-ind2)/2)],[int((matrix_size[2]-ind3)/2),int((matrix_size[2]-ind3)/2)],[0,0]])
    px=tf.pad(x,paddings,'CONSTANT')
    return px

    
def A_H_A(x,ind1,ind2,ind3,matrix_size,D):
    """
    This function is the A^H*A operator as described in paper
    """
    x=tf.dtypes.cast(x,tf.complex64)
    x=x[:,:,:,:,0]

    D=tf.convert_to_tensor(D)
    D=tf.dtypes.cast(D,tf.complex64)
    D=D**2
    #scaling factor
    SF=np.sqrt(matrix_size[0]*matrix_size[1]*matrix_size[2])
    SF=tf.dtypes.cast(tf.convert_to_tensor(SF),tf.complex64)
    x=tf.signal.fft3d(x)/SF
    ty=tf.signal.ifft3d(tf.multiply(D,x))*SF
    #cut to the original size
    y=ty[:,int((matrix_size[0]-ind1)/2):int((matrix_size[0]-ind1)/2)+ind1,int((matrix_size[1]-ind2)/2):int((matrix_size[1]-ind2)/2)+ind2,int((matrix_size[2]-ind3)/2):int((matrix_size[2]-ind3)/2)+ind3]   
    y=tf.expand_dims(y,axis=-1)
    y=tf.dtypes.cast(y,tf.float32)
    return y

def A_op(x,ind1,ind2,ind3,matrix_size,D):
    """
    This function is the A operator (F^-1*D*F) as described in paper
    """
    x=tf.dtypes.cast(x,tf.complex64)
    x=x[:,:,:,:,0]
    
    #dipole kernel
    D=tf.convert_to_tensor(D)
    D=tf.dtypes.cast(D,tf.complex64)
    #scaling factor
    SF=np.sqrt(matrix_size[0]*matrix_size[1]*matrix_size[2])
    SF=tf.dtypes.cast(tf.convert_to_tensor(SF),tf.complex64)
    x=tf.signal.fft3d(x)/SF
    ty=tf.signal.ifft3d(tf.multiply(D,x))*SF
    #cut to the original size
    y=ty[:,int((matrix_size[0]-ind1)/2):int((matrix_size[0]-ind1)/2)+ind1,int((matrix_size[1]-ind2)/2):int((matrix_size[1]-ind2)/2)+ind2,int((matrix_size[2]-ind3)/2):int((matrix_size[2]-ind3)/2)+ind3]   
    y=tf.expand_dims(y,axis=-1)
    y=tf.dtypes.cast(y,tf.float32)
    return y
    

def term2(inputs,ind1,ind2,ind3,matrix_size,D):
    """
    This function performs the term: -t_k A^H A x^k-1 as described in paper  
    """
    x=inputs[0]
    alpha=inputs[1]
    x=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})(x)
    x=Lambda(A_H_A,output_shape=(ind1,ind2,ind3,1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size,'D':D})(x)
    weight2=Lambda(lambda y:-y[1]*y[0])
    x=weight2([x,alpha])
    return x

def term1(y,ind1,ind2,ind3,matrix_size,D):
    """
    This function performs the term: t_k A^H y as described in paper 
    """
    y=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size})(y)
    y=Lambda(A_op,output_shape=(ind1,ind2,ind3,1),arguments={'ind1':ind1,'ind2':ind2,'ind3':ind3,'matrix_size':matrix_size,'D':D})(y)
    return y
    
def init(x):
    """
    This function creates a zero keras tensor of the same size as x
    """
    return tf.zeros_like(x)
        
def My_init(shape, dtype='float32',name=None):
    """
    This function creates the learnable step size in gradient descent    
    """

    value = 4.

    return K.variable(value, name=name)



class MyLayer(Layer):
    """
    This function creates the custom layer in Keras
    """
    
    def __init__(self,**kwargs):
     
        super(MyLayer,self).__init__(**kwargs)
        
    def build(self,input_layer):
        self.step=self.add_weight(name='step',
                                    shape=(1,1),
                                    initializer=My_init,
                                    trainable=True)
        super(MyLayer,self).build(input_layer)
        
    def call(self, input_layer):
        return self.step*tf.ones(tf.shape(input_layer))



      
        
def define_generator(D,is_train,Normdata,input_shape=[48,48,48],output_size=[48,48,48]):
    """
    This function is called to create the generator model
    """
    
    CosTrnMean=Normdata['CosTrnMean']
    CosTrnStd=Normdata['CosTrnStd']

    
    matrix_size=D.shape
    init_input=Input(shape=input_shape+[1])
    conv1=initial_conv(init_input,32,(3,3,3))
    print(conv1.shape)
    wide_res1=Res_block(conv1,k=2,dropout=0.5)
    wide_res2=Res_block(wide_res1,k=2,dropout=0.5)
    wide_res3=Res_block(wide_res2,k=2,dropout=0.5)
    wide_res4=Res_block(wide_res3,k=2,dropout=0.5)
    wide_res5=Res_block(wide_res4,k=2,dropout=0.5)
    wide_res6=Res_block(wide_res5,k=2,dropout=0.5)
    wide_res7=Res_block(wide_res6,k=2,dropout=0.5)
    wide_res8=Res_block(wide_res7,k=2,dropout=0.5)
        
    conv2=initial_conv(wide_res8,32,(1,1,1))
    print(conv2.shape)
    conv3=initial_conv(conv2, 32, (1,1,1))
    output=Conv3D(filters=1,kernel_size=(1,1,1),strides=(1,1,1),padding='same')(conv3)
        
    print(output.shape)
    basic_model=Model(init_input,output)
    
    y_init=Input(shape=input_shape+[1])    #input_2
    mask=Input(shape=input_shape+[1])
    Alpha= MyLayer()(y_init)
    size=tf.keras.backend.int_shape(y_init) 
       
    y_input=Lambda(term1,output_shape=(size[1],size[2],size[3],1),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size,'D':D})(y_init)
    y_input=Multiply()([y_input,Alpha])

    #iterative
    for i in range(3):
        if i==0:
            layer_input=y_input
        else:
            term_output=Lambda(term2,output_shape=(size[1],size[2],size[3],1),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size,'D':D})([x_output,Alpha]) #lamda_9  #lamda_20 #lamda_31
            layer_output=Add()([x_output,term_output])    #add_9  add_11 add_13
            layer_input=Add()([layer_output,y_input])    #add_10    add_12  add_14
            
        layer_input=Lambda(lambda x: (x-CosTrnMean)/CosTrnStd)(layer_input)
        layer_input=Multiply()([layer_input,mask])
        fx_output=basic_model(layer_input)   #model_1  
        fx_output=Multiply()([fx_output,mask])
        x_output=Lambda(lambda x: x*CosTrnStd+CosTrnMean)(fx_output)
  
        

       
                
    final_output=fx_output
    tempx, tempy, tempz = np.round((np.array(input_shape) - np.array(output_size))/2).astype(int)
 
    
    if is_train==1:
        label=Input(shape=output_size+[1])
        phi_genout=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size})(final_output)
        phi_genout=Lambda(A_op,output_shape=(size[1],size[2],size[3],1),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size,'D':D})(phi_genout)
        cropped_phi_genout=Cropping3D(((tempx, tempx), (tempy, tempy), (tempz, tempz)))(phi_genout)
        cropped_y_init=Cropping3D(((tempx, tempx), (tempy, tempy), (tempz, tempz)))(y_init)
        
        Data_consistency_loss=Lambda(lambda x:0.5*K.mean(K.square(x[1]-x[0]),axis=-1))([cropped_y_init,cropped_phi_genout])
        
        final_output  = Cropping3D(((tempx, tempx), (tempy, tempy), (tempz, tempz)))(final_output)
        L1_loss=Lambda(lambda x:K.mean(K.abs(x[1]-x[0]),axis=-1))([label,final_output])
        Tot_loss=Add()([Data_consistency_loss,L1_loss])
       
        model=Model([y_init,mask,label],Tot_loss)
        
        model.compile(optimizer=Adam(lr=0.001,beta_1=0.5),loss=lambda y_true,y_pred:y_pred)
        return model
    else:
        
        final_output  = Cropping3D(((tempx, tempx), (tempy, tempy), (tempz, tempz)))(final_output)
        model=Model([y_init,mask],final_output)
        return model

    
    
def define_discriminator(input_shape=[48,48,48]):
    """
    This function is called to create the discriminator model
    """
    init_input=Input(shape=input_shape+[1])
    conv1=Conv3D(filters=16,kernel_size=(4,4,4),strides=(1,1,1),padding='same')(init_input) 
    conv1=Conv3D(filters=16,kernel_size=(4,4,4),strides=(2,2,2),padding='same')(conv1) 
    conv1=LeakyReLU(alpha=0.2)(conv1)  
    
    conv2=Conv3D(filters=32,kernel_size=(4,4,4),strides=(1,1,1),padding='same')(conv1) 
    conv2=Conv3D(filters=32,kernel_size=(4,4,4),strides=(2,2,2),padding='same')(conv2) 
    conv2=LeakyReLU(alpha=0.2)(conv2) 
    
    conv3=Conv3D(filters=64,kernel_size=(4,4,4),strides=(1,1,1),padding='same')(conv2) 
    conv3=Conv3D(filters=64,kernel_size=(4,4,4),strides=(2,2,2),padding='same')(conv3) 
 #   conv3 = BatchNormalization(axis=4)(conv3)
    conv3=LeakyReLU(alpha=0.2)(conv3) 
    
    conv4=Conv3D(filters=128,kernel_size=(4,4,4),strides=(1,1,1),padding='same')(conv3) 
    conv4=Conv3D(filters=128,kernel_size=(4,4,4),strides=(2,2,2),padding='same')(conv4) 
 #   conv4 = BatchNormalization(axis=4)(conv4)
    conv4=LeakyReLU(alpha=0.2)(conv4)
    
    output=Conv3D(filters=1,kernel_size=(3,3,3),strides=(1,1,1),padding='valid')(conv4) 
    output=Flatten()(output)
    print(output.shape) 
    
    model= Model(init_input,output) 
    model.compile(loss='mse',optimizer=Adam(lr=1e-5,beta_1=0.5))
    return model

def define_gan(D,Normdata,generator,discriminator,input_shape=[48,48,48]):
    """
    This function is called to create the GAN model for training
    """   
    frozen_D=Model(inputs=discriminator.inputs,outputs=discriminator.outputs)
    frozen_D.trainable=False
    discriminator.trainable=False
    CosTrnMean=Normdata['CosTrnMean']
    CosTrnStd=Normdata['CosTrnStd']
    matrix_size=D.shape
    init_input=Input(shape=input_shape+[1])
    mask=Input(shape=input_shape+[1])
    label=Input(shape=input_shape+[1])
    size=tf.keras.backend.int_shape(init_input) 
    
    genout=generator([init_input,mask])
    inorm_genout=Lambda(lambda x: x*CosTrnStd+CosTrnMean)(genout)
    phi_genout=Lambda(pad_tensor,output_shape=(matrix_size[0],matrix_size[1],matrix_size[2],1),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size})(inorm_genout)
    phi_genout=Lambda(A_op,output_shape=(size[1],size[2],size[3],1),arguments={'ind1':size[1],'ind2':size[2],'ind3':size[3],'matrix_size':matrix_size,'D':D})(phi_genout)
    Data_consistency_loss=Lambda(lambda x:K.mean(K.square(x[1]-x[0]),axis=-1))([init_input,phi_genout])
    L1_loss=Lambda(lambda x:K.mean(K.abs(x[1]-x[0]),axis=-1))([label,genout])
    
   # disout=discriminator(genout)
    disout=frozen_D(genout)

    LS_loss=Lambda(lambda x:0.01*K.mean(K.square(1-x),axis=-1))(disout)
    Total_loss=Add()([Data_consistency_loss,L1_loss])
    Total_loss=Add()([Total_loss,LS_loss])
    model=Model([init_input,mask,label],Total_loss)
    model.compile(optimizer=Adam(lr=1e-4,beta_1=0.5),loss=lambda y_true,y_pred:y_pred)
    return model
           
        



