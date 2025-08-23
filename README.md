Orchid Machine Learning Models

Hello! In this repository you will find files related to my senior research project:
Designing a CNN and ViT to identify differing hydration levels in a custom dataset of orchid images.
I worked on this project at Connecticut College with my reseach advisor Professor Timothy Becker from Janurary - December 2024.

Dataset:
I created a custom dataset for this project, as what little data exists for orchid hydration images are private.
I took care of 6 orchids in the Connecticut College greehouse, keeping a control group, an overhydrated group, and an underhydrated group.
Control: Given 200mL of water 2x/week.
Overhydrated: Given with 200mL of water daily.
Underhydrated: Given with 200mL of water 1x/2 weeks.

8 pictures were taken of the leaves from different rotations each day on a black staging area I made from carboard and felt.
Eventually the orchids were transferred to clear pots (vented to prevent root rot) so that images of the roots could be taken, also 8x daily from different rotations.
I rotated each image as well so that the avilable data was octupled and the model could be more robust to variation.
I split the data into 70% training data, 20% validation data, and 10% testing data.

Results:
After much experimentation, the architecture and the values in the files yielded the best results for me.
I acheived accuracy in the mid-70% range. I was able to correct much of the overfitting, particularly with data augmentation.
A lot of the spots where the ViT gets confused is between the control group and the other two groups, since it is the most similar to both of them.

Future Work:
I graduated in December 2024 and haven't worked on the project since except to upload this, but in addition to cleaning up some of the files to make them a little neater and more user-friendly, here are some ideas for the future of the project:
-Integrating the ViT and the CNN
-Experiment more with the image resolution
-Expand data augmentation (lighting, more intense zooms, etc.)

The dataset used for this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/towlealexander/orchid-hydration-data-for-machine-learning-models).
Place the folder in the same directory as the rest of the files.

More detailed instructions are given in the comments of each of the files, but as an overview:

photo_sorting.py - Splits photos into training, validation, and testing data, based on what percentage you want in each group.
In the orchid_data folder, there is a HERO7 BLACK folder and inside that there are three folders called unprocessed_control, unprocessed_overhydrated, and unprocessed_underhydrated.
Place unprocessed images of each group into these folders and run photo_sorting.py to split into the training, validation, and testing sets.

cnn.py and cnn.ipynb - Contains code that trains the CNN and produces results and visualizations. Also does data augmentation before training.

starting_vit_model.py and starting_vit_model.ipynb - Contains the code that trains the ViT and produces results and visualizations. Also does data augmentation before training.

requirements.txt - Requirements, can run with pip install <requirement>.
