from torch.utils.data import Dataset
import numpy as np
from utils.augmentation import augment_sample

# labels are defined as: background=0, liver=1, unlabeled=2
UNLABELED = 2

class UpdatableDataset(Dataset):
    def __init__(self, X, augment=False, Normalization = False):
        self.X = X # shape: (samples_count, 1, image_slices, *image_shape)
        
        # initialize all labels of all slices with unlabeled
        self.y = np.full_like(X, UNLABELED, dtype=np.float32)
        self.augment = augment
        self.Normalization = Normalization
    
        self.max_zoom = 1 # if `max_zoom`=0.9, the zoomed image will contain at least 0.9 of the original image

        self.rotate_max_angle = 1 # random rotation with random angle in range (`-rotate_max_angle`, `rotate_max_angle`)

        # deformation options
        self.deform_points = 4
        self.deform_min_distort = 0.
        self.deform_max_distort = 3.
    
    def __len__(self):
        return len(self.X) # number of samples
        
    def new_label(self, index, slice_num, new_label): # merges a generated label with the previous one
        # index: index of the sample
        # slice_num: the slice number that the label belongs to
        # new_label: the new generated label. shape: (*image_shape)
        
        # find which labels to replace with the new one
        voxels_to_replace = np.logical_and(
            self.y[index, 0, slice_num] == UNLABELED,
            new_label != UNLABELED
        )
        
        # replace
        self.y[index, 0, slice_num][voxels_to_replace] = new_label[voxels_to_replace]
        
        # return the voxels that have been affected.
        return voxels_to_replace
    
    def count_labeled_voxels(self):
        return np.sum(self.y != UNLABELED)
    
    def get_class_weights(self):
        eps = 1e-6
        ones = np.sum(self.y == 1)
        zeros = np.sum(self.y == 0)
        
        return [ones / (ones + zeros + eps), zeros / (ones + zeros + eps)]
    
    def __getitem__(self, index): # sampling method. used by DataLoader.
        sample_X, sample_y = self.X[index], self.y[index]
        if self.augment:
            cond = sample_y[0, :, 0, 0] != UNLABELED
            sample_X[:, cond, :, :], sample_y[:, cond, :, :] = augment_sample(
                sample_X[:, cond, :, :], 
                sample_y[:, cond, :, :],
                max_zoom=self.max_zoom,
                rotate_max_angle=self.rotate_max_angle,
                deform_points=self.deform_points,
                deform_max_distort=self.deform_max_distort,
                deform_min_distort=self.deform_min_distort
            )
            if self.Normalization:
                sample_X = (sample_X - sample_X.min()) / (sample_X.max() - sample_X.min())
                
        return sample_X, sample_y, index # we return the index as well for future use
    
    
    
    
    
    
