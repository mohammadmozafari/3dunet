import skimage.transform
import numpy as np
import cv2

def deform(image_list, points, max_distort, min_distort=0.):
    # create deformation grid 
    rows, cols = image_list[0].shape[0], image_list[0].shape[1]
    src_cols = np.linspace(0, cols, points)
    src_rows = np.linspace(0, rows, points)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add distortion to coordinates
    s = src[:, 1].shape
    dst_rows = src[:, 1] + np.random.normal(size=s)*np.random.uniform(min_distort, max_distort, size=s)
    dst_cols = src[:, 0] + np.random.normal(size=s)*np.random.uniform(min_distort, max_distort, size=s)
    
    dst = np.vstack([dst_cols, dst_rows]).T

    tform = skimage.transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = rows 
    out_cols = cols
    
    return np.array(
        [skimage.transform.warp(image, tform, output_shape=(out_rows, out_cols), mode="symmetric") for image in image_list]
    )

def random_zoom(images, zoom):
    width, height = images[0].shape[0], images[0].shape[1]
    new_image_width, new_image_height = int(width * zoom), int(height * zoom)
    
    if new_image_width == width and new_image_height == new_image_height:
        return images
    
    w_start = np.random.randint(0, width - new_image_width)
    w_end = w_start + new_image_width
    
    h_start = np.random.randint(0, height - new_image_height)
    h_end = h_start + new_image_height
    
    return np.array([cv2.resize(img[w_start:w_end, h_start:h_end], (width, height)) for img in images])

def augment_sample(sample_X, sample_y, max_zoom, rotate_max_angle, deform_points, deform_max_distort, deform_min_distort=0.):
    # sample_X: X of a 3d sample. shape: (1, image_slices, width, height)
    # sample_y: label of a 3d sample: (image_slices, width, height)
    
    # drop the channel's dimension
    sample_X = sample_X[0, :, :, :]
    sample_y = sample_y[0, :, :, :]
    
    # concat all and apply random deformation
    aug = deform(np.concatenate([sample_X, sample_y], axis=0), deform_points, deform_max_distort, deform_min_distort)
    #aug = elasticdeform.deform_random_grid(np.concatenate([sample_X, sample_y], axis=0), 3, 4)
    
    # random zoom
    zoom = np.random.uniform(max_zoom, 1.)
    aug = random_zoom(aug, zoom=zoom)
    
    # split x`s and y`s
    aug_X = aug[:len(sample_X)]
    aug_y = aug[len(sample_X):]
    
    # rotation
    random_angle = np.random.randint(-rotate_max_angle, rotate_max_angle)
    aug_X = np.array([skimage.transform.rotate(x, random_angle, mode='symmetric') for x in aug_X])
    aug_y = np.array([skimage.transform.rotate(y, random_angle, mode='symmetric') for y in aug_y])
    
    # add channel's dimension
    aug_X = np.expand_dims(aug_X, axis=0)
    aug_y = np.expand_dims(aug_y, axis=0)

    aug_y[aug_y < 1.52590219e-05] = 0.
    aug_y[aug_y > 1.52590219e-05] = 1.
    
    return aug_X, aug_y
