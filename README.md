# Capstone Project - Monsoon 2025

### Pipeline

1. Train binary classification model using ResNet.
2. Annotate Data using VGG Annotator and convert the labelling to YOLO format.
3. Exploring SAM for labelling
4. Train YOLO model for multiclass classification.

#### ResNet training

Using the tags put by a colleague(through DigiKam), I am going to train a ResNet model to do the classification. 

Command to access metadata of the image using exiftool:

``` exiftool image.jpg ```

- Place all raw images in images folder.
- Use ```migrateImages.py``` to transfer the images to dataset folder
- Use ```imageProcessing.ipynb``` to extract categories of the images
- Use ```createLabels``` to generate binary labels from the extracted categories
- use model training to finally train the model

- Add script to see incorrectly identified images
- make web app to make it usable

### sherlock documentation:
https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14254





See how many have animals and how many have empty in both training and validation set.

Run the model on all images, do random checks on 10% images of the entire dataset, give result of this to Pooja.

minimizing false negative is essential

in the dataset, get statistics on how many animals of each category

look at it like a binary classification, tiger or not tiger

tiger and only emtpy frames
tiger and only other animal frames

remind professor early next week