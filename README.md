## YOLOV5: Implementation of the You Only Look Once Object Detection Model in PyTorch (version 5.0 in Ultralytics)
---

## Required Environment
torch==1.2.0

## Training Steps
### a. Train VOC07+12 Dataset
1. Dataset Preparation  
   **This article uses the VOC format for training. Before training, download the VOC07+12 dataset, unzip it, and place it in the root directory.**

2. Dataset Processing  
   Modify the annotation_mode to 2 in voc_annotation.py, then run voc_annotation.py to generate 2007_train.txt and 2007_val.txt in the root directory.

3. Start Network Training  
   The default parameters in train.py are for training the VOC dataset. Simply run train.py to start training.

4. Predict Training Results  
   Predicting training results requires two files: yolo.py and predict.py. First, modify the model_path and classes_path in yolo.py – these parameters must be modified.  
   **model_path points to the trained weight file in the logs folder.  
   classes_path points to the txt file corresponding to the detection classes.**  
   After modification, run predict.py for detection. Enter the image path when prompted.

### b. Train Your Own Dataset
1. Dataset Preparation  
   **This article uses the VOC format for training. Before training, create your own dataset.**  
   Place label files in the Annotation folder under VOCdevkit/VOC2007.  
   Place image files in the JPEGImages folder under VOCdevkit/VOC2007.

2. Dataset Processing  
   After arranging the dataset, use voc_annotation.py to obtain 2007_train.txt and 2007_val.txt for training.  
   Modify the parameters in voc_annotation.py. For the first training, you can only modify classes_path, which points to the txt file corresponding to the detection classes.  
   When training your own dataset, create a cls_classes.txt with the classes you want to distinguish.  
   The content of model_data/cls_classes.txt is:
   ```
   ```
   Modify classes_path in voc_annotation.py to correspond to cls_classes.txt and run voc_annotation.py.

3. Start Network Training  
   **There are many training parameters in train.py. After downloading the library, carefully read the comments. The most important part is still the classes_path in train.py.**  
   **classes_path points to the txt file corresponding to the detection classes, the same as in voc_annotation.py! It must be modified for training your own dataset!**  
   After modifying classes_path, run train.py to start training. After multiple epochs, weights will be generated in the logs folder.

4. Predict Training Results  
   Predicting training results requires two files: yolo.py and predict.py. In yolo.py, modify model_path and classes_path.  
   **model_path points to the trained weight file in the logs folder.  
   classes_path points to the txt file corresponding to the detection classes.**  
   After modification, run predict.py for detection. Enter the image path when prompted.

## Prediction Steps
1. Set options in predict.py for FPS testing and video detection.  
###  Use Trained Weights from Your Own Training
