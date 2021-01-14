# -*- coding: utf-8 -*-
"""
This file contains some supporting functions used during testing.

Created on Wed Oct 21 11:20:03 2020

@author: frm
"""

import numpy as np
import nibabel as nib
from numpy.fft import fftn,fftshift,ifftn
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from model.MoG_QSM import * 
from scipy.io import loadmat

def data_predict(model_name, input_data,mask_data, input_data_patch_shape, output_data_patch_shape):
    """
    This function divides the input data into patches and feeds them to the network for testing
    """    
    shift=36;
    temp_X, temp_Y, temp_Z = input_data.shape
    temp_x, temp_y, temp_z = output_data_patch_shape
    temp_xi, temp_yi, temp_zi = input_data_patch_shape
    temp_px, temp_py, temp_pz = int((temp_xi-temp_x)/2), int((temp_yi-temp_x)/2), int((temp_zi-temp_z)/2) 
    temp_pad_x, temp_pad_y, temp_pad_z = int(np.ceil((temp_X-temp_x)/shift)*shift-(temp_X-temp_x)), int(np.ceil((temp_Y-temp_y)/shift)*shift-(temp_Y-temp_y)), int(np.ceil((temp_Z-temp_z)/shift)*shift-(temp_Z-temp_z))
    input_data_pad = np.pad(input_data, ((0, temp_pad_x), (0, temp_pad_y), (0, temp_pad_z)), 'edge')
    mask_data_pad = np.pad(mask_data, ((0, temp_pad_x), (0, temp_pad_y), (0, temp_pad_z)), 'edge')
    
    temp_Xp, temp_Yp, temp_Zp = input_data_pad.shape
    output = np.zeros_like(input_data_pad)
    input_data_pad = np.pad(input_data_pad, ((temp_px, temp_px), (temp_py, temp_py), (temp_pz, temp_pz)), 'edge')
    mask_data_pad = np.pad(mask_data_pad, ((temp_px, temp_px), (temp_py, temp_py), (temp_pz, temp_pz)), 'edge')
    
    output_patches = []
    num_k=int((temp_Zp-temp_z)/shift+1)
    num_j=int((temp_Yp-temp_y)/shift+1)
    num_i=int((temp_Xp-temp_x)/shift+1)
    for k in range(num_k):
        for j in range(num_j):
            for i in range(num_i):
                input_data_patch = input_data_pad[shift*i:(shift*i+temp_xi), shift*j:(shift*j+temp_yi), shift*k:(shift*k+temp_zi)]
                mask_data_patch = mask_data_pad[shift*i:(shift*i+temp_xi), shift*j:(shift*j+temp_yi), shift*k:(shift*k+temp_zi)]
                input_data_patch = np.expand_dims(input_data_patch, axis=0)
                input_data_patch = np.expand_dims(input_data_patch, axis=-1)
                mask_data_patch = np.expand_dims(mask_data_patch, axis=0)
                mask_data_patch = np.expand_dims(mask_data_patch, axis=-1)
                output_patch = model_name.predict([input_data_patch,mask_data_patch])
                temp_patch=output_patch[0,:,:,:,0]
                                         
                if i!=0:
                    patch2=output_patches[-1]
                    temp_patch=patch_process(temp_patch,patch2,output_data_patch_shape[0]-shift,0)
                if j!=0:
                    patch2=output_patches[-num_i]
                    temp_patch=patch_process(temp_patch,patch2,output_data_patch_shape[1]-shift,1)
                if k!=0:
                    patch2=output_patches[-num_i*num_j]
                    temp_patch=patch_process(temp_patch,patch2,output_data_patch_shape[2]-shift,2)
                    
                output[shift*i:(shift*i+temp_x),shift*j:(shift*j+temp_y),shift*k:(shift*k+temp_z)]=temp_patch
               
                output_patches.append(temp_patch)
                
    outputs_data = output[:temp_X, :temp_Y, :temp_Z]
    output_patches_data = np.array(output_patches)
    return outputs_data, output_patches_data



def patch_process(patch1,patch2,overlap,direction):
    """
    This function is called to stitch the image patches 
    patch1: patch to be stitched
    patch2: the patch overlapped with patch1
    direction: The direction of stitching
    """       
    size1=patch1.shape
    size2=patch2.shape
    result=patch1
    if size1==size2:
        size=size1
    else:
        raise ValueError("two patches don't have the same size")

             
    weight=np.ones(overlap) 
    for x in range(int(overlap)):
        weight[x]=1-x/(overlap-1)
            
            
    if direction==0:
        block1=patch1[0:overlap,:,:]
        block2=patch2[int(size[0]-overlap):size[0],:,:]
        for i in range(int(overlap)):
            result[i,:,:]=weight[i]*block2[i,:,:]+(1-weight[i])*block1[i,:,:]
    
    elif direction==1:
        block1=patch1[:,0:overlap,:]
        block2=patch2[:,int(size[1]-overlap):size[1],:]
        for j in range(int(overlap)):
            result[:,j,:]=weight[j]*block2[:,j,:]+(1-weight[j])*block1[:,j,:]
            
    elif direction==2:
        block1=patch1[:,:,0:overlap]
        block2=patch2[:,:,int(size[2]-overlap):size[2]]
        for k in range(int(overlap)):
            result[:,:,k]=weight[k]*block2[:,:,k]+(1-weight[k])*block1[:,:,k]
    else:
        raise ValueError("direction must be a integer between 0 and 2")
            
    return result

def dipole_kernel(matrix_size,voxel_size,B0_dir):
    """
    This function generates the dipole kernel in QSM physical model
    """
    Y,X,Z=np.meshgrid(np.arange(-matrix_size[1]/2,matrix_size[1]/2),np.arange(-matrix_size[0]/2,matrix_size[0]/2),np.arange(-matrix_size[2]/2,matrix_size[2]/2))
    
    X=X/(matrix_size[0]*voxel_size[0])
    Y=Y/(matrix_size[1]*voxel_size[1])
    Z=Z/(matrix_size[2]*voxel_size[2])
    
    D=1/3-(X*B0_dir[0]+Y*B0_dir[1]+Z*B0_dir[2])**2/(X**2+Y**2+Z**2)
    D[np.isnan(D)]=0
    D=np.fft.fftshift(D)
    return D

def inter_data(ori_data,orivox,newvox):
    """
    This function performs interpolation by padding the kspace
    """
    ksp=fftshift(fftn(fftshift(ori_data)));
    N=np.array(ori_data.shape)
    FOV=N*np.array(orivox)   
    newN=np.ceil(FOV/newvox)   
    for i in range(0,3):
        if np.mod(newN[i],2)==1:
            newN[i]=newN[i]-1
    px, py, pz = int(abs((N[0]-newN[0])/2)), int(abs((N[1]-newN[1])/2)), int(abs((N[2]-newN[2])/2))   
    if newN[0]!=N[0]:
        if newN[0]>N[0]:
            ksp=np.pad(ksp, ((px, px), (0, 0), (0, 0)), 'constant')
        else:
            ksp=ksp[px:N[0]-px,:,:]
    if newN[1]!=N[1]:
        if newN[1]>N[1]:
            ksp=np.pad(ksp, ((0, 0), (py, py), (0, 0)), 'constant')
        else:
            ksp=ksp[:,py:N[1]-py,:]
    if newN[2]!=N[2]:
        if newN[2]>N[2]:
            ksp=np.pad(ksp, ((0, 0), (0, 0), (pz, pz)), 'constant')
        else:
            ksp=ksp[:,:,pz:N[2]-pz]    
    return np.real(fftshift(ifftn(fftshift(ksp))))
    

def save_nii(arr,path,voxel_size=[1,1,1]):
    affine=np.array([[voxel_size[0],0,0,0],
                     [0,voxel_size[1],0,0],
                     [0,0,voxel_size[2],0],
                     [0,0,0,1]])
    nib.Nifti1Image(arr,affine).to_filename(path)
    return None

def crop_array(data):
    """
    This function is called to crop a 3D-array to an even size
    """
    size=data.shape
    if np.mod(size[0],2)==1:
        data=np.delete(data,0,0)
    if np.mod(size[1],2)==1:
        data=np.delete(data,0,1)
    if np.mod(size[2],2)==1:
        data=np.delete(data,0,2)
    return data

def change_array_size(data,newsize):
    """
    This function is called to change the size of an array by padding or cropping to the new size
    """
    dim=data.shape
    if dim[0]!=newsize[0] or dim[1]!=newsize[1] or dim[2]!=newsize[2]:
        if dim[0]!=newsize[0]:
            pad_size=int(abs((dim[0]-newsize[0])/2))
            if dim[0]<newsize[0]:
                data=np.pad(data, ((pad_size, pad_size), (0, 0), (0, 0)), 'constant')
            else:
                data=data[pad_size:dim[0]-pad_size,:,:]
        if dim[1]!=newsize[1]:
            pad_size=int(abs((dim[1]-newsize[1])/2))
            if dim[1]<newsize[1]:
                data=np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)), 'constant')
            else:
                data=data[:,pad_size:dim[1]-pad_size,:]
        if dim[2]!=newsize[2]:
            pad_size=int(abs((dim[2]-newsize[2])/2))
            if dim[2]<newsize[2]:
                data=np.pad(data, ((0, 0), (0, 0), (pad_size, pad_size)), 'constant')
            else:
                data=data[:,:,pad_size:dim[2]-pad_size]
    return data        
                          
def model_test(model_dir,phi_data,mask_data,voxel_size,B0_dir):
    """
    This is the main function for testing
    model_dir: directory of the trained model
    phi_data:input normalized tissue phase
    mask_data:input mask data
    voxel_size: voxel size of the input phase
    B0_dir:B0 direction
    
    """
    Normdata=loadmat('../NormFactor.mat')   
    CosTrnMean=Normdata['CosTrnMean']
    CosTrnStd=Normdata['CosTrnStd']   
    if sum(np.mod(np.array(phi_data.shape),2))!=0:
            phi_data=crop_array(phi_data)
            mask_data=crop_array(mask_data)    
    if voxel_size[0]!=1 or voxel_size[1]!=1 or voxel_size[2]!=1:
        inter_phi=inter_data(phi_data,voxel_size,[1,1,1])
        matrix_size=np.array(inter_phi.shape)
        D=dipole_kernel(matrix_size,[1,1,1],B0_dir)
        g_model=define_generator(D,0,Normdata,input_shape=[48,48,48],output_size=[48,48,48])
        g_model.load_weights(model_dir)
        [Y_data,output_3Dpatch]=data_predict(g_model,inter_phi,np.ones_like(inter_phi), [48,48,48], [48,48,48])
        Y_data=Y_data*CosTrnStd+CosTrnMean;
        Y_data=inter_data(Y_data,[1,1,1],voxel_size)
        Y_data=change_array_size(Y_data,mask_data.shape)*mask_data
    else:
        matrix_size=np.array(phi_data.shape)
        D=dipole_kernel(matrix_size,[1,1,1],B0_dir)
        g_model=define_generator(D,0,Normdata,input_shape=[48,48,48],output_size=[48,48,48])
        g_model.load_weights(model_dir)
        [Y_data,output_3Dpatch]=data_predict(g_model,phi_data,mask_data, [48,48,48], [48,48,48])
        Y_data=Y_data*CosTrnStd+CosTrnMean;
        Y_data=Y_data*mask_data
    return Y_data
    
    