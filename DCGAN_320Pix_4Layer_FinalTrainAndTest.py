# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:23:19 2024

Script is used to establish parametric study for deep convolution generative
adversarial network (DCGAN) with 7 layers for image sizes of 255x255 pixels.

Parametric study is manually conducted and alters the noise vector, number of
epochs, and 


@author: matt9
"""

## IMPORT PACKAGES AND LIBRARIES
import mlflow
import torch
import torch.nn as nn
#import torchvision
from torchinfo import summary
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#import numpy as np
import matplotlib.pyplot as plt
#import itertools
import os
import csv
import pandas as pd
#import PIL
from PIL import Image
#import pdb; pdb.set_trace()


##---------------------------------------------------------------------------##

## Inputs
# Hyperparameters
batch_size_array = [5];
z_size_array = [25]      
lr_array = [0.0001];
beta1_array = [0.90];    #(default is 0.9)
beta2_array = [0.999]; #(default is 0.999)
num_epochs = 150000;
n_filters = 16;  # Don't change. I'm fixing the architecture
pixel_size = 320; # Don't change. I'm fixing the architecture
leaky_slope_array = [0.2]; 
rand_rot = 45; 

experiment_name = "HPC_4Layer_FinalTrainAndTest"
mlflow.set_experiment(experiment_name)

# Image & Sampling Parameters
training_annotations_file = "Normalized_Cropped_SEM_Annotated_Training_Images_For_GAN.csv"
testing_annotations_file = "Normalized_Cropped_SEM_Annotated_Testing_Images_For_GAN.csv"
image_size = (pixel_size,pixel_size)  

# Size of Noise Array (Not Including My Parameters)
mode_z = 'uniform'  # 'uniform' vs 'normal'
num_film_parameters = 2

# Test Cases (Inputs are in range [0,1])
# [1, 1, 3.5, 3.5, 10] wt% // 10 = [0.10, 0.10, 0.35, 0.35, 1.00]
fixed_wt_perc = torch.tensor([[0.10], [0.10], [0.35], [0.35], [1.00]],
                             dtype=torch.float32)
# [0.25k, 25k, 1k, 25k, 50k] // 100k =
#    [0.0025, 0.25, 0.01, 0.25, 0.50] mag
fixed_mag = torch.tensor([[0.0025], [0.25], [0.01], [0.25], [0.50]],
                         dtype=torch.float32)

##---------------------------------------------------------------------------##

## Development Options
# Seeds for Development Purposes
#torch.manual_seed(1)
#np.random.seed(1)

# Set Device Options for HPC vs Local Machine
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    # Data Paths
    training_image_path = ("/share/zikry/mphilli2/NSF_Ribbing/GAN_PyTorch/SEM_Training_Images_For_GAN")
    training_image_annotations_file = os.path.join(
        training_image_path,training_annotations_file)
    testing_image_path = ("/share/zikry/mphilli2/NSF_Ribbing/GAN_PyTorch/SEM_Testing_Images_For_GAN")
    testing_image_annotations_file = os.path.join(
        testing_image_path,testing_annotations_file)
    # Turn Off Interactive Plotting
    plt.ioff()
else:
    device = "cpu"
    # Data Paths
    training_image_path = ("C:\\Users\\matt9\\Documents\\" \
                  "PhD\\Projects\\NSF_Ribbing\\" \
                  "Data\\SEM_ImagesOfCNTPDMS\\SEM_Training_Images_For_GAN")
    training_image_annotations_file = os.path.join(
        training_image_path,training_annotations_file)
    testing_image_path = ("C:\\Users\\matt9\\Documents\\" \
                  "PhD\\Projects\\NSF_Ribbing\\" \
                  "Data\\SEM_ImagesOfCNTPDMS\\SEM_Testing_Images_For_GAN")
    testing_image_annotations_file = os.path.join(
        testing_image_path,testing_annotations_file)

##---------------------------------------------------------------------------##

## Define Generator
def Generator(input_size, n_filters, leaky_slope):
    model = nn.Sequential(
        ## Layer 1: len(z)x1 to 9x9
        nn.ConvTranspose2d(input_size, n_filters*4, 
                           kernel_size=9, 
                           stride=1, 
                           padding=0, 
                           bias=False),
        nn.BatchNorm2d(n_filters*4),
        nn.LeakyReLU(leaky_slope),
        
        ## Layer 2: 9x9 to 33x33
        nn.ConvTranspose2d(n_filters*4, n_filters*2, 
                           kernel_size=9, 
                           stride=3, 
                           padding=0, 
                           bias=False),
        nn.BatchNorm2d(n_filters*2),
        nn.LeakyReLU(leaky_slope),
        
        ## Layer 3: 33x33 to 105x105
        nn.ConvTranspose2d(n_filters*2, n_filters, 
                           kernel_size=9, 
                           stride=3, 
                           padding=0, 
                           bias=False),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(leaky_slope),
        
        ## Layer 4: 105x105 to 320x320
        nn.ConvTranspose2d(n_filters, 1, 
                           kernel_size=8, 
                           stride=3, 
                           padding=0, 
                           bias=False),
        nn.Tanh()
    )
    return model


## Define Generator
class Discriminator(nn.Module):
    def __init__(self, n_filters, leaky_slope):
        super().__init__()
        self.network = nn.Sequential(
            ## Layer 1: 320x320 to 105x105
            nn.Conv2d(1, n_filters, 
                      kernel_size=8, 
                      stride=3, 
                      padding=0, 
                      bias = False),
            nn.LeakyReLU(leaky_slope),
            
            ## Layer 2: 105x105 to 33x33
            nn.Conv2d(n_filters, n_filters*2,
                      kernel_size=9, 
                      stride=3, 
                      padding=0, 
                      bias = False),
            nn.BatchNorm2d(n_filters*2),
            nn.LeakyReLU(leaky_slope),
            
            ## Layer 3: 33x33 to 9x9
            nn.Conv2d(n_filters*2, n_filters*4,
                      kernel_size=9, 
                      stride=3, 
                      padding=0, 
                      bias = False),
            nn.BatchNorm2d(n_filters*4),
            nn.LeakyReLU(leaky_slope),
            
            ## Layer 4: 9x9 x 1x1
            nn.Conv2d(n_filters*4, 1,
                      kernel_size=9, 
                      stride=1, 
                      padding=0, 
                      bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        output = self.network(input)
        return output.view(-1,1).squeeze(0)
        

## Define Noise Sampler
def Noise(batch_size,z_size,mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size,z_size, 1, 1)*2-1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size,z_size, 1, 1)
    return input_z

##---------------------------------------------------------------------------##

## Create or Load Training Annotation File
if os.path.isfile(training_image_annotations_file) is True:
    print("Found Training Image Annotation File\n")
else:    
    print("Writing Training Image Annotation File\n")
    x_max = 10;
    y_max = 100000;
    with open(training_image_annotations_file, 'w',newline='') as f:
        writer = csv.writer(f)       
        for x in os.listdir(path=training_image_path):
            if os.path.isdir(os.path.join(training_image_path,x)):
                x_num = float(x.split('wt%')[0])/x_max
                for y in os.listdir(path=os.path.join(training_image_path,x)):
                    _, _, tmpfiles = next(
                        os.walk(os.path.join(training_image_path,x,y)))
                    y_num = float(y.split('x')[0])/y_max
                    for z in os.listdir(path=os.path.join(
                            training_image_path,x,y,"Cropped")):
                        row = [os.path.join(x,y,"Cropped",z), x_num, y_num]
                        writer.writerow(row)
            

## Create or Load Testing Annotation File
if os.path.isfile(testing_image_annotations_file) is True:
    print("Found Testing Image Annotation File\n")
else:    
    print("Writing Testing Image Annotation File\n")
    x_max = 10;
    y_max = 100000;
    with open(testing_image_annotations_file, 'w',newline='') as f:
        writer = csv.writer(f)       
        for x in os.listdir(path=testing_image_path):
            if os.path.isdir(os.path.join(testing_image_path,x)):
                x_num = float(x.split('wt%')[0])/x_max
                for y in os.listdir(path=os.path.join(testing_image_path,x)):
                    _, _, tmpfiles = next(
                        os.walk(os.path.join(testing_image_path,x,y)))
                    y_num = float(y.split('x')[0])/y_max
                    for z in os.listdir(path=os.path.join(
                            testing_image_path,x,y,"Cropped")):
                        row = [os.path.join(x,y,"Cropped",z), x_num, y_num]
                        writer.writerow(row)
            

## Define Custom Image Extraction Routine
class CustomImageDataset(Dataset):
    def __init__(self, image_annotations_file, img_dir, transform, 
                 target_transform=None):
        self.img_labels = pd.read_csv(image_annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        weight_percent = self.img_labels.iloc[idx, 1]
        magnification = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            weight_percent = self.target_transform(weight_percent)
            magnification = self.target_transform(magnification)
        return image, weight_percent, magnification

##---------------------------------------------------------------------------##

## Begin Parametric Study
for z_size in z_size_array:
    ## Initialize Inputs for Test Samples
    fixed_z = Noise(5,z_size,mode_z).to(device)
    fixed_wt_perc = torch.reshape(fixed_wt_perc,(5,1,1,1)).to(device)
    fixed_mag = torch.reshape(fixed_mag,(5,1,1,1)).to(device)
    fixed_z = torch.cat((fixed_z,fixed_wt_perc,fixed_mag),1)
    
    for batch_size in batch_size_array:
        for lr in lr_array:
            for beta1 in beta1_array:
                for beta2 in beta2_array:
                    for leaky_slope in leaky_slope_array:
                        
                        # Define generator and discriminator models
                        g_model = Generator(
                            z_size+num_film_parameters,
                            n_filters,leaky_slope).to(device)
                        d_model = Discriminator(
                            n_filters,leaky_slope).to(device)
                        
                        # Define binary cross entropy loss function
                        loss_fn = nn.BCELoss() 
                        
                        # Define Adam's optimizer for models
                        g_optimizer  = torch.optim.Adam(
                            g_model.parameters(),lr=lr,betas=(beta1,beta2))
                        d_optimizer  = torch.optim.Adam(
                            d_model.parameters(),lr=lr,betas=(beta1,beta2))
                        
                        # Define discriminator training routine
                        def d_train(x,weight_percent,magnification):
                            d_model.zero_grad()
                            
                            batch_size = x.size(0)
                            x = x.to(device)
                            weight_percent = weight_percent.to(device)
                            magnification = magnification.to(device)
                            
                            d_labels_real = torch.ones(
                                batch_size,1,device=device)
                            d_prob_real = d_model(x)
                            d_loss_real = loss_fn(d_prob_real,d_labels_real)
                            
                            input_z = Noise(
                                batch_size,z_size,mode_z).to(device)
                            input_z = torch.cat(
                                (input_z,weight_percent,magnification),1)
                            #breakpoint()
                            
                            g_output = g_model(input_z)
                            d_prob_fake = d_model(g_output)
                            d_labels_fake = torch.zeros(
                                batch_size,1,device=device)
                            d_loss_fake = loss_fn(d_prob_fake,d_labels_fake)
                            
                            d_loss = d_loss_real + d_loss_fake
                            d_loss.backward()
                            d_optimizer.step()                            
                            return d_loss.data.item(), d_prob_real.detach(), \
                                d_prob_fake.detach()
                                
                        # Define generator training routine
                        def g_train(x,weight_percent,magnification):
                            g_model.zero_grad()
                            batch_size = x.size(0)
                            weight_percent = weight_percent.to(device)
                            magnification = magnification.to(device)
                            
                            # Concatenate Film Parameters to Input Array
                            input_z = Noise(batch_size,z_size,
                                            mode_z).to(device)
                            input_z = torch.cat((input_z,weight_percent,
                                                 magnification),1).to(device)
                            input_z = input_z.type(torch.float32).to(device)
                            g_labels_real = torch.ones(
                                batch_size,1,device=device)
                            #breakpoint()
                            
                            g_output = g_model(input_z)
                            d_prob_fake = d_model(g_output)
                            g_loss = loss_fn(d_prob_fake,g_labels_real)
                            
                            g_loss.backward()
                            g_optimizer.step()
                            return g_loss.data.item()
                        
                        
                        # Define discriminator testing routine
                        def d_test(x,weight_percent,magnification):
                            
                            # Test Discriminator Against Real Images
                            test_size = x.size(0)
                            x = x.to(device)
                            weight_percent = weight_percent.to(device)
                            magnification = magnification.to(device)
                            
                            d_labels_real = torch.ones(
                                test_size,1,device=device)
                            d_test_prob_real = d_model(x)
                            d_test_loss_real = \
                                loss_fn(d_test_prob_real,d_labels_real)
                            
                            # Test Discriminator Against Generated Images
                            input_z = Noise(
                                test_size,z_size,mode_z).to(device)
                            input_z = torch.cat(
                                (input_z,weight_percent,magnification),1)
                            #breakpoint()
                            
                            g_output = g_model(input_z)
                            d_test_prob_fake = d_model(g_output)
                            d_labels_fake = torch.zeros(
                                test_size,1,device=device)
                            d_test_loss_fake = \
                                loss_fn(d_test_prob_fake,d_labels_fake)
                            
                            d_test_loss = d_test_loss_real + d_test_loss_fake
                            d_test_loss.backward()                            
                            return d_test_loss.data.item(), \
                                d_test_loss_real.data.item(), \
                                d_test_loss_fake.data.item()
                        
                        # Define sampling routine
                        def create_sample(g_model,input_z):
                            g_output = g_model(input_z)
                            images = torch.reshape(
                                g_output,(batch_size,*image_size))
                            return (images+1)/2.0
                        
                        # Begin MLflow tracking 
                        with mlflow.start_run(nested=True):
                            run = mlflow.active_run()
                            print(f"Active run_id: {run.info.run_id}")
                            
                            params = {
                                "epochs": num_epochs,
                                "learning_rate": lr,
                                "beta1": beta1,
                                "beta2": beta2,
                                "leaky_slope": leaky_slope,
                                "batch_size": batch_size,
                                "noise_size": z_size,
                                "n_filters": n_filters,
                                "pixel_size": pixel_size,
                                "loss_function": loss_fn.__class__.__name__,
                                "optimizer": "Adam",
                                "rand_rotation": rand_rot,
                            }
                            # Log training parameters.
                            mlflow.log_params(params)
                            
                            # Log model summary
                            with open("g_model_summary.txt", 
                                      "w", encoding='utf-8') as f:
                                f.write(str(summary(g_model)))
                            mlflow.log_artifact("g_model_summary.txt")
                            
                            with open("d_model_summary.txt", 
                                      "w", encoding='utf-8') as f:
                                f.write(str(summary(d_model)))
                            mlflow.log_artifact("d_model_summary.txt")
                            
                            epoch_samples = []
                            all_d_losses = []
                            all_g_losses = []
                            all_d_real = []
                            all_d_fake = []
                            all_d_test_losses = []
                            all_d_test_real = []
                            all_d_test_fake = []
                            
                            headers = ['Epoch','G Losses','D Losses', \
                                       'Test D Losses', 'Test D Real', \
                                           'Test D Fake'];
                            f = open('DCGAN_Mean_Losses.csv', 'w')
                            # create the csv writer
                            writer = csv.writer(f)
                            # write a row to the csv file
                            writer.writerow(headers)
                            # close the file
                            f.close()
                            
                            # Begin model training
                            for epoch in range(1,num_epochs+1):
                                # Define Transform for Images
                                transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(rand_rot),
                                    transforms.RandomCrop(image_size),
                                    transforms.Grayscale(),
                                    transforms.Normalize(
                                        mean=(0.5), std=(0.5))])
                                
                                
                                # Load Images to Dataset
                                training_dataset = CustomImageDataset(
                                    training_image_annotations_file, 
                                    training_image_path, 
                                    transform)                                
                                training_dataloader = \
                                    DataLoader(
                                        training_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

                                testing_dataset = CustomImageDataset(
                                    testing_image_annotations_file, 
                                    testing_image_path, 
                                    transform)       
                                testing_size = \
                                    len(testing_dataset.img_labels)+1;
                                testing_dataloader = \
                                    DataLoader(
                                        testing_dataset,
                                        batch_size=testing_size,
                                        shuffle=True)
                                
                                # Begin batch training
                                g_model.train()         # Check me
                                d_losses, g_losses = [], []
                                for i, (x, weight_percent, magnification) \
                                    in enumerate(training_dataloader):
                                    
                                        # Reshape wt% & Magx
                                    weight_percent = \
                                        weight_percent.view(len(x),-1)
                                    magnification = \
                                        magnification.view(len(x),-1)
                                    weight_perc = torch.reshape(
                                        weight_percent,(len(x),1,1,1))
                                    weight_perc_float = weight_perc.type(
                                        torch.FloatTensor)
                                    mag = torch.reshape(
                                        magnification,(len(x),1,1,1))
                                    mag_float = mag.type(torch.FloatTensor)
                                    #breakpoint()
                                    
                                    #breakpoint()
                                    d_loss, d_prob_real, d_prob_fake = d_train(
                                        x,weight_perc_float,mag_float)
                                    d_losses.append(d_loss)
                                    g_losses.append(
                                        g_train(x,weight_perc_float,mag_float))
                                
                                g_loss_mean = torch.FloatTensor(
                                    g_losses).mean();
                                d_loss_mean = torch.FloatTensor(
                                    d_losses).mean();
                                
                                d_test_losses = []
                                d_test_real_losses = []
                                d_test_fake_losses = []
                                ## Evaluate Testing Errors
                                for i, (x, weight_percent, magnification) \
                                    in enumerate(testing_dataloader):
                                    
                                        # Reshape wt% & Magx
                                    weight_percent = \
                                        weight_percent.view(len(x),-1)
                                    magnification = \
                                        magnification.view(len(x),-1)
                                    weight_perc = torch.reshape(
                                        weight_percent,(len(x),1,1,1))
                                    weight_perc_float = weight_perc.type(
                                        torch.FloatTensor)
                                    mag = torch.reshape(
                                        magnification,(len(x),1,1,1))
                                    mag_float = mag.type(torch.FloatTensor)
                                    #breakpoint()
                                    
                                    #breakpoint()
                                    d_test_loss, d_test_loss_real, \
                                        d_test_loss_fake = d_test(
                                            x,weight_perc_float,mag_float)
                                    d_test_losses.append(d_test_loss)
                                    d_test_real_losses.append(d_test_loss_real)
                                    d_test_fake_losses.append(d_test_loss_fake)
                                    
                                d_test_loss_mean = torch.FloatTensor(
                                    d_test_losses).mean();
                                d_test_real_loss_mean = torch.FloatTensor(
                                    d_test_real_losses).mean();
                                d_test_fake_loss_mean = torch.FloatTensor(
                                    d_test_fake_losses).mean();                                
                                
                                
                                print(f'Epoch {epoch:04d} | Avg Lossess >>'
                                      f' G/D {g_loss_mean:.4f}'
                                      f'/{d_loss_mean:.4f}')
                                
                                
                                all_g_losses.append(g_loss_mean)
                                all_d_losses.append(d_loss_mean)
                                all_d_test_losses.append(d_test_loss_mean)
                                all_d_test_real.append(d_test_real_loss_mean)
                                all_d_test_fake.append(d_test_fake_loss_mean)
                                
                                mlflow.log_metric(
                                    "d_loss", f"{d_loss_mean:.4f}", 
                                    step=epoch)
                                mlflow.log_metric(
                                    "g_loss", f"{g_loss_mean:.4f}", 
                                    step=epoch)
                                mlflow.log_metric(
                                    "d_test_loss", f"{d_test_loss_mean:.4f}", 
                                    step=epoch)
                                mlflow.log_metric(
                                    "d_test_real_loss", 
                                    f"{d_test_real_loss_mean:.4f}", 
                                    step=epoch)
                                mlflow.log_metric(
                                    "d_test_fake_loss", 
                                    f"{d_test_fake_loss_mean:.4f}", 
                                    step=epoch)
                                
                                f = open('DCGAN_Mean_Losses.csv', 'a')
                                # create the csv writer
                                writer = csv.writer(f)
                                # write a row to the csv file
                                writer.writerow(
                                    [epoch, 
                                     g_loss_mean.numpy(), 
                                     d_loss_mean.numpy(),
                                     d_test_loss_mean.numpy(),
                                     d_test_real_loss_mean.numpy(),
                                     d_test_fake_loss_mean.numpy()])
                                # close the file
                                f.close()
                                
                                g_model.eval()
                                
                                epoch_samples.append(
                                    create_sample(
                                        g_model,fixed_z).detach().cpu().numpy()
                                )
                            
                            mlflow.pytorch.log_model(d_model, "d_model")
                            mlflow.pytorch.log_model(g_model, "g_model")
                            
                            selected_epochs = [1, 
                                               num_epochs // 20, 
                                               num_epochs // 10, 
                                               num_epochs // 5,
                                               num_epochs // 2, 
                                               num_epochs];
                            fig = plt.figure(figsize=(10,14))
                            for i,e in enumerate(selected_epochs):
                                for j in range(5):
                                    ax = fig.add_subplot(6, 5, i*5+j+1)
                                    ax.set_xticks([])
                                    ax.set_yticks([])
                                    if j == 0:
                                        ax.text(-0.06,0.5, f'Epoch {e}',
                                            rotation=90, size=18, color='red',
                                            horizontalalignment='right',
                                            verticalalignment='center',
                                            transform=ax.transAxes
                                        )
                                    image = epoch_samples[e-1][j]
                                    ax.imshow(image,cmap='gray_r')
                            
                            if torch.cuda.is_available():
                                mlflow.log_figure(
                                    fig, 'DCGAN_Predictions.png')
                                mlflow.log_artifact(
                                    'DCGAN_Mean_Losses.csv')
                            else:
                                #plt.show()
                                mlflow.log_figure(
                                    fig, 'DCGAN_Predictions.png')
                                mlflow.log_artifact(
                                    'DCGAN_Mean_Losses.csv')
