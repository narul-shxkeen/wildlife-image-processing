# Capstone Project - Monsoon 2025

### Pipeline

1. Train binary classification model using ResNet.
2. Annotate Data using VGG Annotator and convert the labelling to YOLO format.
3. Train YOLO model for multiclass classification.

#### ResNet training

Using the tags put by a colleague(through DigiKam), I am going to train a ResNet model to do the classification. 

Command to access metadata of the image using exiftool:

``` exiftool image.jpg ```

- Place all raw images in images folder.
- Use ```migrateImages.py``` to transfer the images to dataset folder
- Use ```imageProcessing.ipynb``` to extract categories of the images
- Use ```createLabels``` to generate binary labels from the extracted categories
- use model training to finally train the model