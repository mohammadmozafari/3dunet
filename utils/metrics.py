import torch
from torch.nn import functional as F
from utils.dataset import UNLABELED

smooth = 1e-2 # smooth value for dice_coef calculation

def dice_coef(outputs, target):
    # target: labels of current batch. can contain UNLABELED or probabilities.
    #     shape: (batch_size, 1, image_slices, width, height)
    
    # outputs: outputs of the model for current batch, without any activations.
    #     shape: (batch_size, 1, image_slices, width, height)
    
    # get the batch size
    batch_size = target.shape[0]
    
    # we calcaulate dice for each sample separately and return their sum.
    if batch_size > 1:
        result = dice_coef(outputs[0:1], target[0:1]) # dice of the first sample in batch
        for i in range(1, len(target)): # add dice of the other samples
            result = result + dice_coef(outputs[i:i+1], target[i:i+1])
        return result
    
    # convert `outputs` to probability
    outputs = torch.sigmoid(outputs)
    
    # round the probabilities to the closest integer
    target = torch.round(target)
    outputs = torch.round(outputs)
    
    # shape of target and outputs will be (batch_size, 1, image_slices, width, height) here.
    #     containing the class of each voxel.
    
    # find the actual labeled voxels. 
    labeled_voxels = target != UNLABELED
    target = target[labeled_voxels]
    outputs = outputs[labeled_voxels]
    
    # so now both target and outputs have only values 0 and 1 for background and liver respectively.
    
    # calculate intersection of target and outputs for each sample in batch
    intersection = (target * outputs).view(batch_size, -1).sum(-1).float()
    
    # calculate sum of target and outputs for each sample in batch
    union = (target + outputs).view(batch_size, -1).sum(-1).float()

    # numerator of dice_coef
    numerator = 2. * intersection + smooth

    # denominator of dice_coef
    denominator = union + smooth

    # calculate dice for each sample in batch
    coef = numerator / denominator
    
    # sum over samples in batch
    return coef.sum(0)


def soft_binray_cross_entropy(outputs, target, class_weights, voxel_weights=None):
    # target: labels of current batch. can contain UNLABELED or probabilities.
    #     shape: (batch_size, 1, image_slices, width, height)
    
    # outputs: outputs of the model for current batch, without any activations.
    #     shape: (batch_size, 1, image_slices, width, height)
    
    # class_weights: weights of classes. a tensor with shape (2, ). can't be None.
    
    # voxel_weights: a tensor with the exact shape of target's, containing a weight for each voxel. can be set to None.
    #     shape: (batch_size, 1, image_slices, width, height)
    
    # get the batch size
    batch_size = target.shape[0]
    
    # we calcaulate loss for each sample separately and return their sum.
    if batch_size > 1:
        # loss of the first sample in batch
        result = soft_binray_cross_entropy(
            outputs[0:1], 
            target[0:1], 
            class_weights, 
            None if voxel_weights is None else voxel_weights[0:1]
        )
        
        for i in range(1, len(target)): # add loss of the other samples
            result = result + soft_binray_cross_entropy(
                outputs[i:i+1], 
                target[i:i+1], 
                class_weights, 
                None if voxel_weights is None else voxel_weights[i:i+1]
            )
        return result
    
    # max(x, 0) - x * z + log(1 + exp(-abs(x)))
    outputs_positives = outputs.clone()
    outputs_positives[outputs_positives < 0] = 0.
    losses = outputs_positives - outputs * target + torch.log(1 + torch.exp(-torch.abs(outputs)))
    
    # calculate strict target (round the probabilities to closest integers)
    strict_target = torch.round(target)
    
    # initialize weights with zeros
    weights = torch.zeros_like(losses).to(target.device).float()
    
    # we set the weights of voxels according to their class
    # the weight of voxels with class `UNLABELED` will remain zero
    weights[strict_target == 0] = class_weights[0]
    weights[strict_target == 1] = class_weights[1]
    
    # apply voxel_weights if not None
    if voxel_weights is not None:
        weights = weights * voxel_weights
    
    # weighted mean
    return (losses * weights).sum() / weights.sum()