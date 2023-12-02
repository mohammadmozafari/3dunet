# IMPORTS #########################################################################################################
import os

import copy
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pickle
import tqdm
import cv2

os.environ['VXM_BACKEND'] = 'pytorch'

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pathlib import Path
from utils.model import UNet3D
from utils.metrics import dice_coef, soft_binray_cross_entropy
from utils.dataset import UpdatableDataset
from skimage.transform import radon, resize

np.random.seed(0)


# use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PARAMETERS #########################################################################################################

# (1) spleen   *
# (2) right kidney *
# (3) left kidney *
# (4) gallbladder  
# (6) liver  *
# (7) stomach
# (8) aorta
# (11) pancreas  *


# CHAOS => 266 => 272
# Sliver => 394
# 3DIrcadb => 260

ORGAN_idx = 255
ORGAN = 'liver'

# for training a Semi Supervised 3D UNet method
MODEL_NAME = 'CHAOS_Semi'  # 'Fully' (for Fuly), 'Semi' (for Semi)
available_labels_selection = 'largest' # 'all' (for Fuly), 'middle' (for Semi), 'largest' (for Semi)

learning_rate = 1e-1
dropout = 0.                 # drop rate
weight_decay = 0.            # weight decay of the optimizer
model_depth = 4              # the depth of the model
optim_name = 'adam' 

Normalization = True
epochs = 500
batch_size = 1
augment_count = 0
image_shape = (256, 256)
image_slices = 272
n_train = 20
n_valid = 10
n_test = 0
#(8*16) # number of slices to use from each sample. should be divisible by (2 ** depth of network)

###############################################################################################
# use hyper parameters to name the results folder

result_folder_name = MODEL_NAME + '_' + available_labels_selection + '_' + ORGAN
print('result_folder_name: {}'.format(result_folder_name))

## LOAD DATA #########################################################################
# from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

# data_paths = get_data_paths()
# outputs_path = get_outputs_path()

# print(data_paths['data1'])

# saving_base = Experiment().get_outputs_path()
# print(saving_base)

# data_base = data_paths['data1'] + '/SynapsData/medical-data/'

# # path to the preprocessed data folder
# DIM = 512
# labeled_images = np.load(data_base + 'synaps/labeled_images.npy', allow_pickle=True)
# print(len(labeled_images))  

# path = '/home/mozafari/medical-data/raw/synaps/labeled_images.npy'
path = '/home/mozafari/medical-data/raw/3Dircadb.npy'
# path = '/home/mozafari/medical-data/raw/CHAOS-CT-Liver.npy'
# path = '/home/mozafari/medical-data/raw/Sliver07-Liver-Train.npy'

saving_base = 'exps'
labeled_images = np.load(path, allow_pickle=True)
print(len(labeled_images))

x = [i['image'].shape[0] for i in labeled_images]
print(x)
print(max(x))
exit()

# DATA Proceesing ############################################################################################
# black slices are appended to each sample to make the number of slices 208.
# so for each sample, we have a number of real slices and a number of fake slices.
# it's notable that the fake slices are appended at the end of real slices.
# so if a sample has 198 real slices, the first 198 are real and the next 10 are fake.

data_X = []
data_Y = []
for idx in range(len(labeled_images)):
    xx = labeled_images[idx].get("image")
    yy = labeled_images[idx].get("label")
    
    yy[np.where(yy != ORGAN_idx)] = 0
    yy[np.where(yy == ORGAN_idx)] = 1
    
    x = []
    y = []
    for i in range(len(xx)):
        x.append(cv2.resize(xx[i,:,:], image_shape))
        y.append(cv2.resize(yy[i,:,:], image_shape))
    x = np.asarray(x)
    y = np.asarray(y)
        
    if x.shape[0]<image_slices:
        x = np.expand_dims(np.concatenate([x, np.zeros((image_slices - len(x), *x.shape[1:])).astype('int16')], axis=0)
                       , axis=0)
        y = np.expand_dims(np.concatenate([y, np.zeros((image_slices - len(y), *y.shape[1:])).astype('uint8')], axis=0)
                       , axis=0)
    else:
        x = x[x.shape[0] - image_slices:]
        x = x[np.newaxis]
        y = y[y.shape[0] - image_slices:]
        y = y[np.newaxis]
    
    data_X.append(x)
    data_Y.append(y)

data_X = np.asarray(data_X)
data_Y = np.asarray(data_Y)
print(data_X.shape)
print(data_Y.shape)
print(np.unique(data_Y))

train_X = data_X[:n_train]
train_Y = data_Y[:n_train]

# valid_X = data_X[n_train:n_valid+n_train]
# valid_Y = data_Y[n_train:n_valid+n_train]
valid_X = data_X[:n_train]
valid_Y = data_Y[:n_train]

test_X = data_X[n_train+n_valid:n_train+n_valid+n_test]
test_Y = data_Y[n_train+n_valid:n_train+n_valid+n_test]

print(train_X.shape, train_Y.shape)
print(valid_X.shape, valid_Y.shape)

def generate_pivots(samples_count, valid_ranges):
    # for each sample, we select middle slice to label.
    # select a pivot slice for each sample randomly

    pivots = np.array([np.floor_divide(valid_ranges[i][1] - valid_ranges[i][0], 2) + valid_ranges[i][0]
                       for i in range(samples_count)])
    return pivots


def get_valid_range(sample_y):
    start, stop = 0, image_slices
    for i in range(image_slices):
        y = sample_y[:, i, :, :]
        if y.sum() > 0:
            start = i
            break

    for i in range(start, image_slices):
        y = sample_y[:, i, :, :]
        if y.sum() == 0:
            stop = i - 1
            break
    if stop ==0:
        stop = image_slices-1
    return start, stop

valid_ranges = [get_valid_range(y) for y in train_Y]

if available_labels_selection == 'middle':
    # consider the middle real slice of each sample as it's available label.
    available_labels = np.expand_dims(generate_pivots(len(train_Y), valid_ranges), axis=-1)
    
if available_labels_selection == 'largest':
    # consider the middle real slice of each sample as it's available label.
    sizes = train_Y.sum(axis=(1, 3, 4))
    pivots = sizes.argmax(axis=1)
    available_labels = np.expand_dims(pivots, axis=-1)
    
elif available_labels_selection == 'all':
    # for each sample, all REAL slices are available.
    available_labels = [np.arange(valid_ranges[i][0], valid_ranges[i][1]+1) for i in range(len(train_X))]

print('available_labels for training data: {}'.format(available_labels))


## Augmentation #############################################################################################################
from utils.augmentation import augment_sample
# augmentation options
max_zoom = 0.9        # if `max_zoom`=0.9, the zoomed image will contain at least 0.9 of the original image
rotate_max_angle = 5  # random rotation with random angle in range (`-rotate_max_angle`, `rotate_max_angle`)
# deformation options
deform_points = 4
deform_min_distort = 0.
deform_max_distort = 3.

if augment_count > 0:
    # keep original samples
    new_train_X = train_X.copy()
    new_train_y = train_Y.copy()
    new_available_labels = available_labels.copy()
    
    for _ in range(augment_count): # append new augmented samples
        new_aug_X = []
        new_aug_y = []
        for index in range(len(train_X)):
            aug_X, aug_y = augment_sample(
                train_X[index], 
                train_Y[index],
                max_zoom=max_zoom,
                rotate_max_angle=rotate_max_angle,
                deform_points=deform_points,
                deform_max_distort=deform_max_distort,
                deform_min_distort=deform_min_distort
            )
            new_aug_X.append(aug_X)
            new_aug_y.append(aug_y)

        new_train_X = np.concatenate([new_train_X, np.array(new_aug_X)], axis=0)
        new_train_y = np.concatenate([new_train_y, np.array(new_aug_y)], axis=0)
        new_available_labels = np.concatenate([new_available_labels, available_labels.copy()], axis=0)
        
    train_X = new_train_X
    train_Y = new_train_y
    available_labels = new_available_labels

samples_count = len(train_X)
val_samples_count = len(valid_X)
test_samples_count = len(test_X)

print(samples_count, val_samples_count, test_samples_count)    
    
## Define DataLoader ##########################################################################################################

#### FOR TRAINING DATA #########################

# it's notable that all of the labels are considered UNLABELED in the
# constructor of the Dataset
train_loader = DataLoader(
    UpdatableDataset(train_X, Normalization=Normalization),
    batch_size=batch_size,
    shuffle=True,
    num_workers=6
)

# for each sample, we add it's available labels to the dataset.
for index, slices in enumerate(available_labels):  # iterate over samples
    for sl in slices:  # iterate over available labels of this sample
        train_loader.dataset.new_label(index, sl, train_Y[index, 0, sl, :, :])  # add the label to the dataset

# It is notable that fake slices will remain UNLABELED.


#### FOR VALIDATION DATA #########################

valid_ranges = [get_valid_range(y) for y in valid_Y]
valid_real_slices = [np.arange(valid_ranges[i][0], valid_ranges[i][1]+1) for i in range(len(valid_X))]
print('valid_real_slices: {}'.format(valid_real_slices))

valid_loader = DataLoader(UpdatableDataset(valid_X, augment=False, Normalization=Normalization), batch_size=batch_size,
                          shuffle=True, num_workers=6)

# add all the real slices to the validation dataset
for index, slices in enumerate(valid_real_slices):  # iterate over samples
    for sl in slices:  # iterate over available labels of this sample
        valid_loader.dataset.new_label(index, sl, valid_Y[index, 0, sl, :, :])  # add to the dataset

#### FOR Test DATA #########################

#valid_ranges = [get_valid_range(y) for y in test_Y]
#test_real_slices = [np.arange(valid_ranges[i][0], valid_ranges[i][1]+1) for i in range(len(test_X))]
#print('test_real_slices: {}'.format(test_real_slices))

#test_loader = DataLoader(UpdatableDataset(test_X, augment=False, Normalization=Normalization), batch_size=batch_size,
#                          shuffle=True, num_workers=6)

#for index, slices in enumerate(test_real_slices):  # iterate over samples
#    for sl in slices:  # iterate over available labels of this sample
#        test_loader.dataset.new_label(index, sl, test_Y[index, 0, sl, :, :])  # add to the dataset


## DEFINE MODEL ############################################################################################

# initialize the model
model = UNet3D(input_dim=1, output_dim=1, depth=model_depth, dropout=dropout)
model.load_state_dict(torch.load(f'./exps/{MODEL_NAME}_{available_labels_selection}_{ORGAN}/best-model'))
model.to(device).float()

# get the class weights from the train dataset. it's [ones / (ones + zeros), zeros / (ones + zeros)]
class_weights = torch.Tensor(train_loader.dataset.get_class_weights()).to(device)

# define the optimizer.
optimizer = None
if optim_name == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optim_name == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


## Train MODEL #######################################################################################################
results_path = saving_base + '/'

# create a folder to store the results
# if not os.path.isdir(results_path):
#     os.mkdir(results_path)
# os.mkdir(results_path + result_folder_name)  # we store the results of the training in this folder

def log(l):
    with open(results_path + result_folder_name + '/log.txt', 'a') as f:
        f.write("%s\n" % l)
    print(l)

# we save losses for both train and validation.
loss_train = []
loss_val = []
#loss_test = []

# we save best model, it's epoch number and it's validation dice.
# the best model will be save in `best_model` file later.
best_val_dice = 0.
best_model_state_dict = None
best_epoch = None


for epoch in range(epochs):  # epochs loop
    running_loss = 0.  # initialize training loss of this epoch with zero
    running_dice = 0.  # initialize training dice of this epoch with zero
    running_val_loss = 0.  # initialize validation loss of this epoch with zero
    running_val_dice = 0.  # initialize validation dice of this epoch with zero
    #running_test_loss = 0.  
    #running_test_dice = 0.  


    # comment for testing
    
    # for i, data in (enumerate(train_loader)):  # iterate over batches
    #     # get a batch
    #     X, Y, index = data

    #     # send the batch to gpu
    #     X, Y = X.to(device).float(), Y.to(device).float()

    #     # zero the gradients
    #     optimizer.zero_grad()

    #     # Forward pass
    #     outputs = model(X)

    #     # calculate the training loss
    #     # the UNLABELED voxels will be ignored in the process of calculating the loss.
    #     loss = soft_binray_cross_entropy(outputs, Y, class_weights=class_weights)

    #     # Backward and optimize
    #     loss.backward()
    #     optimizer.step()

    #     # add the training loss to the running training loss
    #     running_loss += loss.item()

    #     # calculate dice coefficient on this batch and add it to the running training dice.
    #     # the UNLABELED voxels will be ignored in the process of calculating the dice.
    #     running_dice += dice_coef(outputs, Y).item()
        
        
    # if we have at least one validation sample, we perform validation here.
    if val_samples_count > 0:  # check if there is any validation samples
        # switch to evaluation mode
        model.eval()

        with torch.no_grad():  # no gradients are required in the validation time
            for val_i, val_data in enumerate(valid_loader):  # iterate over validation batches
                # get a batch
                val_X, val_Y, val_index = val_data

                # send the batch to gpu
                val_X, val_Y = val_X.to(device).float(), val_Y.to(device).float()

                # forward pass
                val_outputs = model(val_X)

                # calculate the validation loss
                # the UNLABELED voxels will be ignored in the process of calculating the loss.
                val_loss = soft_binray_cross_entropy(val_outputs, val_Y, class_weights=class_weights)

                # add the validation loss to the running validation loss
                running_val_loss += val_loss.item()

                # calculate dice coefficient on this batch and add it to the running validation dice.
                # the UNLABELED voxels will be ignored in the process of calculating the dice.
                running_val_dice += dice_coef(val_outputs, val_Y).item()
                
                   
    #if test_samples_count > 0:  
        #model.eval()
        #with torch.no_grad():  
            #for val_i, val_data in enumerate(test_loader): 
                #val_X, val_Y, val_index = val_data
                #val_X, val_Y = val_X.to(device).float(), val_Y.to(device).float()
                #val_outputs = model(val_X)
                #test_loss = soft_binray_cross_entropy(val_outputs, val_Y, class_weights=class_weights)
                #running_test_loss += test_loss.item()
                #running_test_dice += dice_coef(val_outputs, val_Y).item()

        
            # log and print the losses and dices of this epoch.
        l = 'Epoch %d: loss=%.8f, dice=%.6f, val_loss=%.8f, val_dice=%.6f'%(epoch + 1,running_loss / samples_count,running_dice / samples_count,running_val_loss / val_samples_count,running_val_dice / val_samples_count)
        
        print(l)
        # Comment for training
        exit()
        
        log(l)
        
        # if this epochs is the best one until now, save the model, epoch number and validation dice.
        if running_val_dice >= best_val_dice:  # if it's the best epoch
            # set the best validation dice to current validation dice
            torch.save(model.state_dict(), results_path + result_folder_name + '/model-' + str(epoch+1))
            print('Model is Saved.')
            
            best_val_dice = running_val_dice
            # set the best model to a copy of current model
            best_model_state_dict = copy.deepcopy(model.state_dict())

            # set the best epoch to current epoch number
            best_epoch = epoch

        # switch back to training mode
        model.train()

    # store the losses in loss_train and loss_val
    loss_train.append(running_loss / samples_count)
    loss_val.append(running_val_loss / val_samples_count)
    #loss_test.append(running_test_loss / test_samples_count)

print('Finished Training')


##_________Save the results

# visualization of losses
epoch = np.arange(len(loss_train))  # list of epoch numbers (0, 1, ..., epochs - 1)
loss_train = np.asarray(loss_train)  # list of train loss for each epoch
loss_val = np.asarray(loss_val)  
#loss_test = np.asarray(loss_test)

# log the best epoch if there is a validation data
if val_samples_count > 0:  
    bests_str = 'best epoch: ' + str(best_epoch + 1) + ', best_val_dice: ' + str(best_val_dice / val_samples_count)
    log(bests_str)
    torch.save(best_model_state_dict, results_path + result_folder_name + '/best-model')

np.save('loss_train.npy', loss_train)
np.save('loss_val.npy', loss_val)
#np.save('loss_test.npy', loss_test)

print('Results Saved at', result_folder_name)    
print('Code is Ended.')

