# MRI-Denoising-Model-
Implementation of Residual Encoder–Decoder Wasserstein Generative Adversarial Network

# In One Drive:
Any model titled with a date 19Nov was trained with a data set that only contined PD-w images. 
Any model titled with a date 27Jov was trained with a data set that contains T1-,T2-, and PD-w image.

In each of the model folders, there is an "output_{job ID}" file which is outputed by a cluster job and 
contains all iterative training infromation. 
    Sadly, 19Nov_baseline does not have an output file because I (Jane Corah) accidentally deleted it, but all other model do have the output file showing evidence of training. 

The folder loss contains the generator, discriminator and testing loss as well as image quality metrics. 
The folder results contains the denoised images from the generator.

Any .pth file is a saved model weight. 
Also in each model folder is a .png which is contains the graphed loss curves for the generator, discriminator and training.

Any file that contains "modelJQC_{model description}.py" was used to train the correpsonding model. 
