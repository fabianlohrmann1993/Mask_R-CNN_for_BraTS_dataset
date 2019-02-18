"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python brats.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python brats.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python brats.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python brats.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python brats.py evaluate --dataset=/path/to/coco/ --model=last
"""


import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
import argparse 

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print("ROOT_DIR:",ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")




                    ###CONFIG CLASS###

class BratsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "brats"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    #related to batch size?
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2


    # Number of classes (including background)
    NUM_CLASSES = 1 + 1
    """
    it is 3 classes ! look in load_images() and common sense
    change it!
    """
        

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.0

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    LEARNING_RATE = 0.0001
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 80

    
    BACKBONE = "resnet101"

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20





                            ### DATA CLASSES ###

class BratsDataset(utils.Dataset):
    mode = None
    tumor_type = None

    def preprocess_image(self, image):
        
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.IntensityWindowing(sitk_image, np.percentile(image, 1), np.percentile(image, 99))
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat64 )

        corrected_image = sitk.N4BiasFieldCorrection(sitk_image, sitk_image > 0);
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        # corrected_image = exposure.equalize_hist(corrected_image)
        return corrected_image
        

    def load_image(self, image_id):
        
        """
        Note:
        FLAIR -> Whole
        T2 -> Core
        T1C -> Active (if present)
        
        Load the specified image and return a [H,W,3] Numpy array.
        The 3 layers are not RGB but the 3 scans! 
        
        """
        info = self.image_info[image_id]
        for path in os.listdir(info['path']):
            if self.mode in path:
                image = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
                break
        image = self.preprocess_image(image)
        #checking if image is 3 layers deeps (for RBB or flair, t2, t1c?)
        if image.ndim != 3:
            image = image.astype(np.uint8)
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            #image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        #this checks if image is has a transparency level (alpha) which would be incoded in the 4th row of a vector
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    def load_images(self, data_dir, subset):
        
        """
        Load a subset (batch) of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        print('Reading images')
        # Add classes
        self.add_class("brats", 1, self.tumor_type)
        
        # Train or validation dataset?
        assert subset in ["train", "val"]
        data_dir = os.path.join(data_dir, subset)

        
        i = 0
        for subdir in os.listdir(data_dir):
            indices = self.getIndicesWithTumorPresent(data_dir + "/" + subdir)
            for j in indices:
                self.add_image("brats", image_id=i, path=data_dir + "/" + subdir, ind = j)
                i = i + 1
        
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "brats":
            return info["source"]
        else:
            super(self.__class__).image_reference(self, image_id)
    
    def getIndicesWithTumorPresent(self, directory):
        indicesWithMasks = []
        path = next((s for s in os.listdir(directory) if "seg" in s), None)
        mask = nib.load(directory+"/"+path).get_data()
        for i in range(0,155):
            if np.count_nonzero(mask[:,:,i]) > 0:
                indicesWithMasks.append(i)
        return indicesWithMasks
        

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """        
        for path in os.listdir(self.image_info[image_id]['path']):
            if "seg" in path:
                mask = nib.load(self.image_info[image_id]['path']+"/"+path).get_data()[:,:,self.image_info[image_id]['ind']]
                break

        mask = self.getMask(mask)
        mask = mask.reshape(mask.shape[0], mask.shape[1],1)
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def getMask(self, mask):
        pass

class T2Dataset(BratsDataset):
        
    def __init__(self):
        super().__init__()

        self.mode = "t2"
        self.tumor_type = "core"
    
    def getMask(self, mask):
        mask[mask == 2] = 0
        mask[mask > 0] = 1
        return mask
        


class T1CDataset(BratsDataset):
    def __init__(self):
        super().__init__()
        self.mode = "t1ce"
        self.tumor_type = "active"
    
    def getMask(self, mask):
        mask[mask < 4] = 0
        mask[mask > 0] = 1
        return mask
    


class FlairDataset(BratsDataset):

    
    def __init__(self):
        super().__init__()
        self.mode = "flair"
        self.tumor_type = "whole"
    
    
    def getMask(self, mask):
        mask[mask > 0] = 1
        return mask

class CombinedDataset(BratsDataset):
    
    def load_images(self, data_dir,subset):
        """
        Load a subset (batch) of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        print('Reading images')
        
        # Add classes
        self.add_class("brats", 1, "whole")
        self.add_class("brats", 2, "active")
        self.add_class("brats", 3, "core")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        data_dir = os.path.join(data_dir, subset)
       
        
        i = 0
        
        for subdir in os.listdir(data_dir):
            indices = self.getIndicesWithTumorPresent(data_dir + "/" + subdir)
            for j in indices:
                self.add_image("brats", image_id=i, path=data_dir + "/" + subdir, ind = j)
                i = i + 1
                
    def load_image(self, image_id):

        """
        ## Note:
        # FLAIR -> Whole
        # T2 -> Core
        # T1C -> Active (if present)
        Load the specified image and return a [H,W,3] Numpy array.
        """
        image = np.zeros((240,240,3))
        info = self.image_info[image_id]
        for path in os.listdir(info['path']):
            if "flair" in path:
                image[:,:,0] = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
            elif "t1ce" in path:
                image[:,:,1] = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
                #image[:,:,1] = self.preprocess_image(image[:,:,1])
            elif "t2" in path:
                image[:,:,2] = nib.load(info['path'] + "/" + path).get_data()[:,:,info['ind']]
                #image[:,:,2] = self.preprocess_image(image[:,:,2])
        image = self.preprocess_image(image)
        return image
    
    def load_mask(self, image_id):
      
        for path in os.listdir(self.image_info[image_id]['path']):
            if "seg" in path:
                mask = nib.load(self.image_info[image_id]['path']+"/"+path).get_data()[:,:,self.image_info[image_id]['ind']]
                break

        mask, class_ids = self.getMask(mask)
        return mask.astype(bool), np.asarray(class_ids, dtype=np.float32)
        
    def getMask(self, mask):
        a = []
        class_ids = []
        
        """
        here he reads the mask to establish a workable mask, a whole picture, but i don't get how. Pixelwise?
        """
        
        whole = self.getWholeMask(mask.copy())
        if np.count_nonzero(whole) > 0:
            class_ids.append(1)
            a.append(whole)
            
            
        active = self.getActiveMask(mask.copy())
        if np.count_nonzero(active) > 0:
            class_ids.append(2)
            a.append(active)

            
        core = self.getCoreMask(mask.copy())
        if np.count_nonzero(core) > 0:
            class_ids.append(3)
            a.append(core)
            
        temp = np.array(a)
        temp = np.swapaxes(temp, 0, 2)
        return temp, class_ids
      
    def getWholeMask(self, mask):
        mask[mask > 0] = 1
        return mask
    
    def getActiveMask(self, mask):
        mask[mask < 4] = 0
        mask[mask > 0] = 1
        return mask
        
    def getCoreMask(self, mask):
        mask[mask == 2] = 0
        mask[mask > 0] = 1
        return mask


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = FlairDataset()
    dataset_train.load_images(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FlairDataset()
    dataset_val.load_images(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                layers='4+')
    #if not enough results, change layers='heads'
    
    """
    THE MRI OLD CODE
    
    WHY DOES IT START AT FLAIRDATA?()
    
    def main():
    config = BratsConfig()
    config.display()
    
    #you need to make the HGG data folder into data_dir, val_dir and test_dir !!!
    data_dir = "C:/Users/flohr/PythonProjects/MaskRCNN/Mask_RCNN/datasets/brats/HGG"
    val_dir = "C:/Users/flohr/PythonProjects/MaskRCNN/Mask_RCNN/datasets/brats/HGG_Validation"
    #test_dir = "C:/Users/flohr/PythonProjects/MaskRCNN/Mask_RCNN/datasets/brats/HGG_Testing"

    
    dataset_train = FlairDataset()
    dataset_train.load_images(data_dir)
    dataset_train.prepare()
    
    
    dataset_val = FlairDataset()
    dataset_val.load_images(val_dir)
    dataset_val.prepare()

    """

############################################################
#  Training
############################################################


if __name__ == '__main__':
    

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect tumor.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/brats/dataset/",
                        help='Directory of the brats dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BratsConfig()
    else:
        class InferenceConfig(BratsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_MODEL_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)

    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))




