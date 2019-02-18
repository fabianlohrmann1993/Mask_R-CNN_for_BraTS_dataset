import sys
import os

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print("ROOT_DIR:",ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import brats  


import h5py
import numpy as np
import random
import matplotlib.pyplot as plt
import math

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

try:
    inference_config = brats.InferenceConfig()
    
except:
    inference_config=brats.BratsConfig()
    

test_dir = "C:/Users/flohr/PythonProjects/MaskRCNN/Mask_RCNN/datasets/brats/test"
data_dir = "C:/Users/flohr/PythonProjects/MaskRCNN/Mask_RCNN/datasets/brats/train"


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
                                                            #this could be a potential error source 'mrimath2018063...'
#result_data = np.genfromtxt(MODEL_DIR +'/brats20180922T1937/model_loss_log.csv', delimiter=',')
result_data = h5py.File(MODEL_DIR + '\brats20180922T1937\mask_rcnn_brats_0003.h5', 'r')

plt.figure()


#plt.plot(result_data[:,0], result_data[:,1],label="total loss")
plt.plot(result_data[:,0], result_data[:,4], label = "loss")
plt.plot(result_data[:,0], result_data[:,10], label = "val loss")

plt.xlabel("Epochs") 
plt.ylabel("Loss") 
plt.legend(loc='upper right')
plt.show()

dataset = brats.FlairDataset()
dataset.load_images(test_dir)
dataset.prepare()
print("Testing on " + str(len(dataset.image_info)) + " images")



# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = model.find_last()[1]
# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True) 

# Test on a random image
"""
here starts the diversion
"""

image_id = random.choice(dataset.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, inference_config, 
                           image_id, use_mini_mask=False)
print(dataset.image_info[image_id])
log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

results = model.detect([original_image], verbose=0)
r = results[0]

#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
 #                           dataset.class_names, figsize=(8, 8))
visualize.display_differences(original_image,
                        gt_bbox, gt_class_id, gt_mask,
                        r["rois"], r["class_ids"], r["scores"], r['masks'],
                        dataset.class_names, show_box=False)     

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset.image_ids, 1000)
#print(image_ids)
#print(dataset.image_ids)
APs = []




precision = {}
precision[1] = []

recall = {}
recall[1] = []

dices = []


for image_id in dataset.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    if ((r['masks'] > 0.5).size <= 0 or (gt_mask > 0.5).size <= 0): 
        continue
    if r['masks'].shape[0] is 256:
        score = -math.inf
        for i in range(0, r['masks'].shape[2]):
            if dice(r['masks'][:,:,i:i+1], gt_mask) > score:
                score = dice(r['masks'][:,:,i:i+1], gt_mask)
        dices.append(score)
        """
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                        r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)
        """
            

            #print(jaccards[label])
    #print(AP)      
    #APs.append(AP)



print("Dice Coefficient: " + str(np.mean(np.asarray(dices))))





# move the testing data back
"""
list_imgs = os.listdir(test_dir)
for sub_dir in list_imgs:
    dir_to_move = os.path.join(test_dir, sub_dir)
    shutil.move(dir_to_move, data_dir)
"""
        
        
        
        
        