# MoG-QSM
MoG-QSM: Model-based Generative Adversarial Deep Learning Network for Quantitative Susceptibility Mapping.  
MoG-QSM was proposed by Ruimin Feng and Dr. Hongjiang Wei. It reconstructs high quality susceptibility maps from tissue phase. 


###Environmental Requirements:
1. Python 3.6
2. Tensorflow 1.15.0
3. Keras 2.2.5


###Files descriptions  
MoG-QSM contains the following folders:
1. data: It provides three types of test data: healthy data from Siemens Prisma scanner, multiple sclerosis data, and 2016 QSM Challenge data.
2. logs/last.h5: A file that contains the weights of the trained model
3. model/MoG_QSM.py : This file contains the functions to create the Model-based Generative Adversarial Network proposed in our paper

4. test: It contains test_tools.py and test_demo.py.  
test_tools.py offers some supporting functions for network testing such as image patch stitching, dipole kernel generation, etc. 
test_demo.py shows how to perform network testing with data from the "data" folder

5. train: It contains train_gen.py, train_joint.py and utils.py.  
train_gen.py: This is the code for generator training  
train_joint.py: This is the code for generator and discriminator jointly training  
utils.py: It offers some supporting functions for network training

6. NormFactor.mat: The mean and standard deviation of our training dataset for input normalization.

###Usage  
##Test  
1. You can run test_demo.py directly to test the network performance on the provided data. The results will be in the same directory as the input data
2.  For test on your own data. You can use "model_test" function as shown in test_demo.py files    

##train   
1. If you want to train MoG-QSM by yourself. train_gen.py and train_joint.py can be used as a reference.
