# -*- coding: utf-8 -*-
"""
This file contains the main test code for two Prisma data, one MS data and QSM 2016 challenge data
Run this code and the results will be in the same directory as the input data 

Created on Thu May 21 14:39:20 2020

@author: frm
"""
import numpy as np
from test_tools import *
from scipy.io import loadmat
from scipy.io import savemat
import nibabel as nib
model_dir='../logs/last.h5'

"""prisma test data subject1""" 
data_path='../data/Prisma_data/sub1/' 
phi_data = np.array(nib.load(data_path+'phi.nii').get_fdata())
mask_data = np.array(nib.load(data_path+'mask.nii').get_fdata())
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1])
nib.Nifti1Image(Y_data, nib.load(data_path+'phi.nii').affine).to_filename(data_path+'MoG_QSM_output.nii')

      
"""prisma test data subject2""" 
data_path='../data/Prisma_data/sub2/' 
phi_data = np.array(nib.load(data_path+'phi.nii').get_fdata())
mask_data = np.array(nib.load(data_path+'mask.nii').get_fdata())
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1])
nib.Nifti1Image(Y_data, nib.load(data_path+'phi.nii').affine).to_filename(data_path+'MoG_QSM_output.nii')


"""MS test data""" 
data_path='../data/MS_data/' 
phi_data = np.array(nib.load(data_path+'phi.nii').get_fdata())
mask_data = np.array(nib.load(data_path+'mask.nii').get_fdata())
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1])
nib.Nifti1Image(Y_data, nib.load(data_path+'phi.nii').affine).to_filename(data_path+'MoG_QSM_output.nii')

"""QSM 2016 challenge test data"""
data_path='../data/QSM_challenge_data/' 
phi_data = np.array(nib.load(data_path+'phi.nii').get_fdata())
mask_data = np.array(nib.load(data_path+'mask.nii').get_fdata())
Y_data=model_test(model_dir,phi_data,mask_data,[1,1,1],[0,0,1])
nib.Nifti1Image(Y_data, nib.load(data_path+'phi.nii').affine).to_filename(data_path+'MoG_QSM_output.nii')





