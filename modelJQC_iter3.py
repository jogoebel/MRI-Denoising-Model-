import os
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import VGG19_Weights
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim

from preprocessing_v2 import patch_test_img, merge_test_img
from mixdata_v5 import *
from datetime import datetime
import csv

'''
Version: 0.0.1
Date: 2018-04-01
Structure:
    Generator: Residual connection
        input  -> 32 -> 64 -> 128 -> 256 
        output <- 32 <- 64 <- 128 <——

        encoder layer: Conv3d -> BatchNorm3d -> LeakyReLu
        decoder layer: Conv3D - > Add encoder -> BatchNorm3d -> LeakyReLU(expect last layer is ReLu)
    Disciminator:
        input -> 32 -> 64 -> 128 -> 1
        Except last layer:Conv3d -> BatchNorm3d -> LeakyReLu
        Last layer: Full Connection(no active function)
'''


class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )
        # torch.nn.init.
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU()
        )

        self.deConv1_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.deConv1 = nn.Sequential(
            nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )

        self.deConv2_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.deConv2 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.deConv3_1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.deConv3 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.deConv4_1 = nn.Conv3d(32, 1, kernel_size=3, padding=1)

        self.deConv4 = nn.ReLU()

    def forward(self, input):
        conv1 = self.conv1(input)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        x = self.deConv1_1(conv4)
        x = x + conv3

        deConv1 = self.deConv1(x)

        x = self.deConv2_1(deConv1)
        x = x + conv2
        deConv2 = self.deConv2(x)

        x = self.deConv3_1(deConv2)
        x = x + conv1
        deConv3 = self.deConv3(x)

        x = self.deConv4_1(deConv3)
        x = x + input
        output = self.deConv4(x)

        return output


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            #nn.BatchNorm3d(32), ## reenabled batch normalization 15Nov2024 JQC
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm3d(64),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm3d(128),
            nn.LeakyReLU()
        )

        self.fc = nn.Linear(128 * 6 * 32 * 32, 1)
        # self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x))
        output = self.fc(x)
        # output = self.fc2(x)

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i

        return num_features


class VGG19(nn.Module): ### issue to check, VGG19 class has means hardcoded into it that are associated with RGB images - do I need to change this to match MRI?
    def __init__(self):
        super(VGG19, self).__init__()
        # Use weights parameter instead of pretrained
        vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.feature = vgg19.features

    '''
    input: N*1*D(6)*H*W
    output: N*C*H*W
    '''

    def forward(self, input):
        # Normalize input
        input = input / 16
        depth = input.size(2)
        result = []

        for i in range(depth):
            # Adjust color channels
            x = torch.cat(
                (input[:, :, i, :, :] - 103.939,
                 input[:, :, i, :, :] - 116.779,
                 input[:, :, i, :, :] - 123.68), 1)
            # Pass through feature extractor
            result.append(self.feature(x))

        output = torch.cat(result, dim=1)
        return output



def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class WGAN():   
    def __init__(self, levels, modalities, version, checkpoint_path=None):
         #Hyperparameters
        #Training Parameters
        self.epochs = 30                 # Total number of training epochs
        self.warmup_epochs = 0          # Number of epochs to keep the learning rate constant
        self.batch_size = 110            # Batch size for training
        
        # Learning rate
        self.lr = 5e-6                   # Initial learning rate
        self.gamma_exp = 0.97                # Exponential decay rate
        self.gamma_step = 0.5           # step decay rate
        self.step_size = 4              # how many epochs between step decays

        # Loss function weights
        self.lambda_gp = 10              # Gradient penalty weight for WGAN-GP
        self.lambda_vgg = 0.1           # Weight for perceptual loss (VGG loss)
        self.lambda_d = 1e-3             # Weight for discriminator loss
        self.lambda_mse = 1           # Weight for mean squared error (MSE) loss

        # Discriminator-specific parameters
        self.d_iter = 3                  # Number of discriminator iterations per generator iteration
        
        # Training Data
        self.levels = levels
        self.modalities = modalities 
         
        # Set device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     

        # Define directories 
        self.version = version
        datetime_str = datetime.now().strftime("%Y%b%d_%H%M%S")
        self.loss_dir = f"./{self.version}/{datetime_str}/loss/"  # store all the D/G loss + image metrics
        os.makedirs(self.loss_dir, exist_ok=True)
        
        self.checkpoint_dir = f"./{self.version}/{datetime_str}/checkpoints/"  # store model checkpoints
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        

        self.generator = GeneratorNet().to(self.device)
        self.discriminator = DiscriminatorNet().to(self.device)
        self.vgg19 = VGG19().to(self.device)

        self.G_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

         # adding learning rate scheduler
        # Exponential
        # self.G_scheduler = lr_scheduler.ExponentialLR(self.G_optimizer, gamma=self.gamma_exp)
        # self.D_scheduler = lr_scheduler.ExponentialLR(self.D_optimizer, gamma=self.gamma_exp)
        
        # Step decay: Halve the learning rate every 4 epochs
        self.G_scheduler = lr_scheduler.StepLR(self.G_optimizer, step_size=self.step_size, gamma=self.gamma_step)
        self.D_scheduler = lr_scheduler.StepLR(self.D_optimizer, step_size=self.step_size, gamma=self.gamma_step)
        
        self.G_loss = nn.MSELoss().to(self.device)
        
        self.checkpoint_path = checkpoint_path # if i have a check point with model weights and optimizers I want to continue to change

        print(f"This is model version {self.version}")
        print(f"Beginning training at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
        print(f"Training Notes: Changed to number the discriminator trains per generator to 3")
        
        # Load checkpoint if provided
        self.start_epoch = 0
        if self.checkpoint_path is not None:
            if os.path.exists(self.checkpoint_path):
                self.start_epoch = self.load_checkpoint(self.generator, self.discriminator, self.G_optimizer, self.D_optimizer, self.checkpoint_path)
                print(f"Checkpoint loaded. Resuming training from epoch {self.start_epoch}.")
            else:
                print(f"Warning: Checkpoint path '{self.checkpoint_path}' does not exist. Initializing from scratch.")
                initialize_weights(self.generator, self.discriminator)
        else:
            print("No checkpoint specified. Initializing from scratch.")
            initialize_weights(self.generator, self.discriminator)
            
    def train(self):
        torch.autograd.set_detect_anomaly(True)
        
        # Create dataset and dataloader for training
        print("loading in training data")
        self.dataset = MRIDataset(levels=self.levels, mode="training", modalities=self.modalities, device = self.device)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True)

        # Create dataset and dataloader for validation
        print("loading in validation data")
        self.validDataset = MRIDataset(levels=self.levels, mode="validation", modalities=self.modalities, device = self.device)
        self.validDataloader = DataLoader(
            self.validDataset, batch_size=self.batch_size, shuffle=False)
        
        # create csv files to log G and D loss
        d_loss_file = os.path.join(self.loss_dir, 'discriminator_log.csv')
        g_loss_file = os.path.join(self.loss_dir, 'generator_log.csv')
        
        # check if files exist, if not, create with headers
        if not os.path.isfile(g_loss_file):
            with open(g_loss_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Batch_Index", "Learning_Rate", "Levels", "G_Overall_Loss", "G_MSE_Loss", "G_Adversarial_Loss", "G_VGG_Loss"])

        if not os.path.isfile(d_loss_file):
            with open(d_loss_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Batch_Index", "Learning_Rate", "Levels", "Iteration", "D_Overall_Loss", "D_Real_Adversarial_Loss", "D_Fake_Adversarial_Loss", "D_Gradient_Penalty"])

        # test before the first training period 
        self.test(self.start_epoch)
        
        #number of epochs
            
        for epoch in range(self.start_epoch,self.epochs):
            epoch = epoch+1
            # iterating over the dataset
            print(f"Epoch {epoch}/{self.epochs}")
            timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Epoch start time: {timestr}, Batch Size: {self.batch_size}")
            for batch_index, batch in enumerate(self.dataloader):
                start_time = time.time()
                # if (batch_index % 100 == 0):
                #     self.test(epoch)
                # print("epoch:", epoch, ";batch number:", batch_index, ";D_Loss:", end="")
                free_img = batch["free_img"]
                noised_img = batch["noised_img"]
            
                # Training discriminator
                for iter_i in range(self.d_iter):
                    d_loss, real_adv_loss, fake_adv_loss, grad_penalty = self._train_discriminator(free_img, noised_img)
                    
                    # Join multiple levels into a string format
                    levels_str = "_".join(map(str, self.levels)) 
                    
                    current_lr = self.G_scheduler.get_last_lr()[0]
                    
                    print(f"\tVGG_MSE - lr: {current_lr:.10f}, Levels: {levels_str}, Epoch: {epoch}, batch_index: {batch_index}, iter: {iter_i}, D-Loss: {d_loss:.4f}, Real: {real_adv_loss:.4f}, Fake: {fake_adv_loss:.4f}, GP: {grad_penalty:.4f}")
                    
                    # Log to D_loss CSV file
                    with open(d_loss_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            epoch, batch_index, current_lr, levels_str, iter_i,
                            d_loss, real_adv_loss, fake_adv_loss, grad_penalty
                        ])

                # Training generator
                g_loss, mse_loss, adv_loss, vgg_loss = self._train_generator(free_img, noised_img)

                # Print G-Loss details
                print(f"\tVGG_MSE - lr: {current_lr:.10f}, Levels: {levels_str}, Epoch: {epoch}, batch_index: {batch_index}, G-Loss: {g_loss:.4f}, MSE: {mse_loss:.4f}, Adv: {adv_loss:.4f}, VGG: {vgg_loss:.4f}")
                
                # Log to G_loss CSV file
                with open(g_loss_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, batch_index, current_lr, levels_str,
                        g_loss, mse_loss, adv_loss, vgg_loss
                    ])

            self.checkpoint_path = self.create_checkpoint_path(self.checkpoint_dir,epoch,current_lr)  
            print(self.checkpoint_path)   
            self.save_checkpoint(self.generator, self.discriminator, self.G_optimizer, self.D_optimizer, epoch, self.checkpoint_path)

            # time of epoch
            end_time = time.time()
            print(f"Epoch Duration: {end_time-start_time}")
            
            # test after epoch
            print(f'testing epoch {epoch}')
            self.test(epoch)
            
            # Step the scheduler to start decay only after warmup epochs
            if epoch >= self.warmup_epochs:
                self.G_scheduler.step()
                self.D_scheduler.step()

    def _train_discriminator(self, free_img, noised_img, train=True):
        self.D_optimizer.zero_grad()

        z = (noised_img).to(self.device)
        real_img = (free_img / 4096).to(self.device) # DO we need to change the /4096 value?

        fake_img = self.generator(z)
        real_validity = self.discriminator(real_img)
        fake_validity = self.discriminator(fake_img.data / 4096)
        gradient_penalty = self._calc_gradient_penalty(
            real_img.data, fake_img.data)

        fake_img_val = fake_img /4096
        print(f"Real Image - Min: {real_img.min().item()}, Max: {real_img.max().item()}")
        print(f"Generated Image - Min: {fake_img_val.min().item()}, Max: {fake_img_val.max().item()}")
        
        d_loss = torch.mean(-real_validity) + torch.mean(fake_validity) + \
            self.lambda_gp * gradient_penalty
        if train:
            d_loss.backward()
            # torch.mean(-real_validity).backward()
            # (torch.mean(-real_validity) + torch.mean(fake_validity)).backward()
            # torch.mean(-real_validity).backward()
            # torch.mean(fake_validity).backward()
            self.D_optimizer.step()

        return d_loss.data.item(), torch.mean(-real_validity).cpu().item(), torch.mean(fake_validity).cpu().item(), self.lambda_gp * gradient_penalty.cpu().item()

    def _train_generator(self, free_img, noised_img, train=True):
        z = noised_img.to(self.device)
        real_img = free_img.to(self.device)

        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()
        self.vgg19.zero_grad()

        criterion_mse = nn.MSELoss().to(self.device)
        criterion_vgg= nn.MSELoss().to(self.device)

        fake_img = self.generator(z)
        mse_loss = criterion_mse(fake_img, real_img)
        if train:
            (self.lambda_mse * mse_loss).backward(retain_graph=True)


        feature_fake_vgg = self.vgg19(fake_img)
        feature_real_vgg = self.vgg19(real_img).detach().to(self.device)

        vgg_loss = criterion_vgg(feature_fake_vgg, feature_real_vgg)

        fake_validity = self.discriminator(fake_img / 4096)
        # g_loss = self.lambda_mse * mse_loss + self.lambda_vgg * vgg_loss + self.lambda_d * torch.mean(-fake_validity)
        g_loss =  self.lambda_vgg * vgg_loss + self.lambda_d * torch.mean(-fake_validity)

        if train:
            # (self.lambda_mse * mse_loss).backward()
            g_loss.backward()
            self.G_optimizer.step()
        return g_loss.data.item(), mse_loss.data.item(), torch.mean(-fake_validity).data.item(), vgg_loss.data.item()

    def _calc_gradient_penalty(self, free_img, gen_img):
        batch_size = free_img.size()[0]
        alpha = torch.rand(batch_size, 1).to(self.device)
        alpha = alpha.expand(batch_size, free_img.nelement(
        ) // batch_size).contiguous().view(free_img.size()).float()

        interpolates = (alpha * free_img + (1 - alpha)
                        * gen_img).requires_grad_(True).to(self.device)
        disc_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(batch_size, 1).fill_(1.0).to(self.device)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]

        # gradients = gradients.view(gradients.size(0), -1)
        # print(gradients.size())
        # print(torch.norm(gradients, 2, dim=1).size())

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # print("gradient_penalty: ", gradient_penalty.cpu().item())
        return gradient_penalty

    def test(self, epoch):
        timestr = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(timestr)
        
        # test set
        total_mse_loss = 0
        total_g_loss = 0
        total_d_loss = 0
        total_vgg_loss = 0
        batch_num = 0
        
        # Testing Loss Log File
        testing_loss_file = os.path.join(self.loss_dir, 'testing_log.csv')
        if not os.path.isfile(testing_loss_file):
            with open(testing_loss_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "G_Overall_Loss", "G_MSE_Loss", "G_Adversarial_Loss/D_Overall_Loss", "G_VGG_Loss"])
    
        for batch_index, batch in enumerate(self.validDataloader):
            free_img = batch["free_img"]
            noised_img = batch["noised_img"]

            loss = self._train_generator(free_img, noised_img, train=False)
            # print(loss, end=" ;")
            total_g_loss += loss[0]
            total_mse_loss += loss[1]
            total_d_loss += loss[2]
            total_vgg_loss += loss[3]
            batch_num += 1
            
        mse_loss = total_mse_loss / batch_num
        g_loss = total_g_loss / batch_num
        d_loss = total_d_loss / batch_num
        vgg_loss = total_vgg_loss / batch_num
        print("Epoch: %d lr: %.10f Test Loss: g-loss: %.4f vgg-loss: %.4f mse-loss: %.4f d_loss: %.4f" %
            (epoch, self.G_scheduler.get_last_lr()[0], g_loss, vgg_loss, mse_loss, d_loss))
       
       # Log results
        print(f"Epoch: {epoch}, Test Losses - G-Loss: {g_loss:.4f}, VGG-Loss: {vgg_loss:.4f}, MSE-Loss: {mse_loss:.4f}, D-Loss: {d_loss:.4f}")
        with open(testing_loss_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, g_loss, mse_loss, d_loss, vgg_loss])

        self.compute_quality()

    def compute_metric(free_img, target_img, dynamic_range):
        """ Helper function to compute PSNR, SSIM, and MSE """
        psnr = compare_psnr(free_img, target_img, data_range=dynamic_range)
        ssim = compare_ssim(free_img, target_img, data_range=dynamic_range, multichannel=True)
        mse = compare_nrmse(free_img, target_img)
        return psnr, ssim, mse

    def compute_quality(self):
        # Loop through each noise level in self.levels
        for level in self.levels:
            for modality in self.modalities:
                print(f"Computing quality metrics for noise level {level} and {modality}...")

                # Initialize quality metrics for this level
                psnr1, psnr2, mse1, mse2, ssim1, ssim2 = 0, 0, 0, 0, 0, 0
                _psnr1, _psnr2, _mse1, _mse2, _ssim1, _ssim2 = 0, 0, 0, 0, 0, 0

                # Define file paths for free and noised images for this level
                free_nii_list = list_niigz_imgs(f"./data/dataset/Free/{modality}/validation")
                noised_nii_list = list_niigz_imgs(f"./data/dataset/noise_{level}/{modality}/validation")

                # Check if the lists have the same length
                if len(free_nii_list) != len(noised_nii_list):
                    print("Warning: The lists have different lengths.")

                # Iterate through both lists together
                for free_path, noised_path in zip(free_nii_list, noised_nii_list):
                    free_nii = nib.load(free_path)
                    noised_nii = nib.load(noised_path)

                    free_img = free_nii.get_fdata()[:, :144, :].astype(np.int16)
                    noised_img = noised_nii.get_fdata()[:, :144, :].astype(np.int16)
                    patchs, row, col = patch_test_img(noised_img)
                    denoised_img = merge_test_img(
                        self.denoising(patchs), row, col).astype(np.int16)

                    # Metrics with a fixed data range (4096)
                    psnr1 += compare_psnr(free_img, noised_img, data_range=4096)
                    psnr2 += compare_psnr(free_img, denoised_img, data_range=4096)
                    mse1 += compare_nrmse(free_img, noised_img)
                    mse2 += compare_nrmse(free_img, denoised_img)
                    ssim1 += compare_ssim(free_img, noised_img, data_range=4096, win_size=5, multichannel=True)
                    ssim2 += compare_ssim(free_img, denoised_img, data_range=4096, win_size=5, multichannel=True)

                    # Metrics with a dynamic data range based on max intensity
                    max_value = np.max(free_img)
                    _psnr1 += compare_psnr(free_img, noised_img, data_range=max_value)
                    _psnr2 += compare_psnr(free_img, denoised_img, data_range=max_value)
                    _mse1 += compare_nrmse(free_img, noised_img)
                    _mse2 += compare_nrmse(free_img, denoised_img)
                    _ssim1 += compare_ssim(free_img, noised_img, data_range=max_value, win_size=5, multichannel=True)
                    _ssim2 += compare_ssim(free_img, denoised_img, data_range=max_value, win_size=5, multichannel=True)

                # Average the metrics
                n_images = len(free_nii_list)
                psnr1, psnr2 = psnr1 / n_images, psnr2 / n_images
                mse1, mse2 = mse1 / n_images, mse2 / n_images
                ssim1, ssim2 = ssim1 / n_images, ssim2 / n_images
                _psnr1, _psnr2 = _psnr1 / n_images, _psnr2 / n_images
                _mse1, _mse2 = _mse1 / n_images, _mse2 / n_images
                _ssim1, _ssim2 = _ssim1 / n_images, _ssim2 / n_images

                # Define file paths for storing metrics
                dynamic_metrics_path = os.path.join(self.loss_dir, f"quality_metrics_dynamic_range_level_{level}_modality_{modality}.csv")
                fixed_metrics_path = os.path.join(self.loss_dir, f"quality_metrics_fixed_range_4096_level_{level}_modality_{modality}.csv")

                # Write headers if the files do not already exist
                if not os.path.exists(dynamic_metrics_path):
                    with open(dynamic_metrics_path, "w") as f:
                        f.write("Time,Learning Rate,PSNR (Noisy-Free),PSNR (Denoised-Free),"
                                "SSIM (Noisy-Free),SSIM (Denoised-Free),MSE (Noisy-Free),MSE (Denoised-Free)\n")
                if not os.path.exists(fixed_metrics_path):
                    with open(fixed_metrics_path, "w") as f:
                        f.write("Time,Learning Rate,PSNR (Noisy-Free),PSNR (Denoised-Free),"
                                "SSIM (Noisy-Free),SSIM (Denoised-Free),MSE (Noisy-Free),MSE (Denoised-Free)\n")

                current_lr = self.G_scheduler.get_last_lr()[0]
                
                # Save metrics for the dynamic data range
                with open(dynamic_metrics_path, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')},{current_lr},{_psnr1},{_psnr2},{_ssim1},{_ssim2},{_mse1},{_mse2}\n")

                # Save metrics for the fixed data range (4096)
                with open(fixed_metrics_path, "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')},{current_lr},{psnr1},{psnr2},{ssim1},{ssim2},{mse1},{mse2}\n")

        
    def denoising(self, patchs):
        n, h, w, d = patchs.shape
        denoised_patchs = []
        for i in range(0, n, self.batch_size):
            batch = patchs[i:i + self.batch_size]
            batch_size = batch.shape[0]
            x = np.reshape(batch, (batch_size, 1, w, h, d))
            x = x.transpose(0, 1, 4, 2, 3)
            x = torch.from_numpy(x).float().to(self.device)
            y = self.generator(x)
            denoised_patchs.append(y.cpu().data.numpy())
        # print(len(denoised_patchs))
        denoised_patchs = np.vstack(denoised_patchs)
        # print(denoised_patchs.shape)
        denoised_patchs = np.reshape(denoised_patchs, (n, d, h, w))
        denoised_patchs = denoised_patchs.transpose(0, 2, 3, 1)
        # print(denoised_patchs.shape)
        return denoised_patchs
            
   
    def save_checkpoint(self, generator, discriminator, G_optimizer, D_optimizer, epoch, path):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'G_optimizer_state_dict': G_optimizer.state_dict(),
            'D_optimizer_state_dict': D_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, generator, discriminator, G_optimizer, D_optimizer, path):
        checkpoint = torch.load(path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")
        return start_epoch
    
    def create_checkpoint_path(self, base_dir, epoch, lr):
        # Define the checkpoint filename with specific parameters
        epoch_str = f"e{epoch}" if epoch else ""
        lr_str = f"_lr{lr:.0e}" if lr else ""
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{epoch_str}{lr_str}_{date_str}.pth"
        return os.path.join(base_dir, filename)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used.")
    else:
        print("CUDA is not available. Using CPU.")
        
    levels = [1,7,13]
    modalities =  ["IXI-PD","IXI-T1","IXI-T2"]
    version = "27Nov_iter3"
    wgan = WGAN(levels, modalities, version)  # Pass both level and modality to the WGAN class

    #Training
    wgan.train()

    # Testing
    for level in levels:
        print(f"Testing for noise level {level}...")
        for modality in modalities:

            # Generate list of test images for the current noise level
            noised_nii_list = list_niigz_imgs(f"./data/dataset/noise_{level}/{modality}/testing")

            # Ensure the output directory for this level exists
            output_dir = f"./{version}/result/noise_{level}/{modality}/"  # add version name to results folder here
            os.makedirs(output_dir, exist_ok=True)

            # Process each image in the test set
            for path in noised_nii_list:
                nii_img = nib.load(path)
                file_name = os.path.basename(path)
                x = nii_img.get_fdata()  # Use get_fdata() for compatibility with newer nibabel versions

                # Split image into patches and process each patch
                patches, row, col = patch_test_img(x)
                print(f"Patches shape for {file_name}: {patches.shape}")

                # Denoise the image patches and merge them back into a full image
                denoised_img = merge_test_img(wgan.denoising(patches), row, col)
                denoised_img = denoised_img.astype(np.int16)
                print(f"Denoised image shape for {file_name}: {denoised_img.shape}")

                # Save the denoised image
                denoised_image = nib.Nifti1Image(denoised_img, nii_img.affine, nii_img.header)
                nib.save(denoised_image, f"{output_dir}wgan_vgg_mse_denoised_img_{file_name}")

