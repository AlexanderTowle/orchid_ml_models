#!/usr/bin/env python
# coding: utf-8

# Connected to orchidenv (Python 3.8.18)

# In[26]:


# imports
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_last')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns


# In[27]:


# paths for training, validation, and testing sets
train_path = '/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK/train'
valid_path = '/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK/valid'
test_path = '/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK/test'


# In[28]:


# method to crop an image
def crop(image):
    h, w, _ = image.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    square_image = image[top:top + min_dim, left:left + min_dim]
    return square_image


# In[29]:


# separate into batches of 3000x4000 size images
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (3000,4000), classes = ['control', 'overhydrated', 'underhydrated'], batch_size = 3, class_mode = 'categorical')
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (3000,4000), classes = ['control', 'overhydrated', 'underhydrated'], batch_size = 48, class_mode = 'categorical')
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (3000,4000), classes = ['control', 'overhydrated', 'underhydrated'], batch_size = 37, class_mode = 'categorical')


# In[30]:


# test to show batch shape
x, y = next(train_batches)
print("Shape of the images batch:", x.shape)
print("Shape of the labels batch:", y.shape)


# In[31]:


# test method to display some sample images
def plots(ims, figsize = (12, 6), rows = 1, interp = False, titles = None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize = figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# In[32]:


# get the next batch
imgs, labels = next(train_batches)


# In[33]:


# show the batch plots
plots(imgs, titles = labels)


# In[34]:


# method to augment images
# basically you it takes an image, a list of the factors you want to zoom it by,
# and a list of the angles you want it to turn by. It zooms that amount and then makes a version
# of the zoomed image for each angle.
# returns the augmented images
def augment_image(image, zoom_factors, angles):
    image = crop(image)
    image = image.astype(np.uint8)
    augmented_images = [image]
    height, width = image.shape[:2]

    for angle in angles:
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(image, M, (width, height))

        for zoom in zoom_factors:
            zoomed_img = cv2.resize(rotated_img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
            
            start_x = (zoomed_img.shape[1] - width) // 2
            start_y = (zoomed_img.shape[0] - height) // 2
            cropped_img = zoomed_img[start_y:start_y + height, start_x:start_x + width]

            augmented_images.append(cropped_img)

    return augmented_images


# In[35]:


# method to plot the augmented images
def plot_images(images, titles=None, rows=2):
    cols = len(images) // rows if len(images) % rows == 0 else len(images) // rows + 1
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        plt.subplot(rows, cols, i+1)

        img = images[i]

        plt.imshow(img)
        plt.axis('off')
        if titles is not None:
            plt.title(titles[i], fontsize=10)
    plt.show()


# In[36]:


#test to show augmentation
zoom_factors = [1.5]
angles = [45, 90, 135, 180, 225, 270, 315]

x_batch, y_batch = next(train_batches)

sample_image = x_batch[0]

augmented_images = augment_image(sample_image, zoom_factors, angles)

plot_images(augmented_images, titles=["Original"] + [f"Zoom {z}, Rotate {a}" for z in zoom_factors for a in [0] + angles])

print(f"Original image min: {sample_image.min()}, max: {sample_image.max()}")

for i, img in enumerate(augmented_images):
    print(f"Augmented image {i} min: {img.min()}, max: {img.max()}")


# In[37]:


# actual class labels, angles, and zoom factors
class_labels = ['control', 'overhydrated', 'underhydrated']

zoom_factors = [1.5]
angles = [45, 90, 135, 180, 225, 270, 315]


# In[38]:


# method to augment and save all image batches into a new folder
global_image_counter = 0

def augment_and_save_batches(batches, save_dir, steps):
    global global_image_counter
    
    for step in range(steps):
        batch = next(batches)
        images, labels = batch[0], batch[1]
        print(f"Processing batch {step + 1}/{steps}...")

        for i in range(len(images)):
            img = images[i]
            label_index = np.argmax(labels[i])
            label_name = class_labels[label_index]

            augmented_imgs = augment_image(img, zoom_factors, angles)

            for j, aug_img in enumerate(augmented_imgs):
                output_path = os.path.join(save_dir, label_name, f'{label_name}_aug_{global_image_counter}_{i}_{j}.jpg')
                
                global_image_counter += 1
                
                cv2.imwrite(output_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))


# In[39]:


# paths where augmented images will be/are saved
train_mod_path = '/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK/train_mod'
valid_mod_path = '/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK/valid_mod'
test_mod_path = '/Users/alextowle/orchids/COM Research Seminar/orchid_data/HERO7 BLACK/test_mod'


# In[53]:


# augment and save the batches
# ONLY DO THIS IF
# 1. YOU HAVE NEW IMAGES IN THE NON-MOD FOLDERS THAT NEED TO BE AUGMENTED (DELETE EXISTING DATA IN MOD FOLDERS FIRST)
# OR
# 2. THERE IS NOTHING IN THE MOD FOLDERS TO BEGIN WITH (YOU WILL NEED TO RUN THIS THE FIRST TIME SINCE I DIDN'T HAVE ENOUGH STORAGE TO SEND ALL THE MOD DATA)
# what i do is i clear out the mod folders when new images need to be added and augmented and then rerun this process
# not the most sophisticated approach i know but i had to work with what i did before i knew i needed augmentation
augment_and_save_batches(train_batches, train_mod_path, steps=502)
augment_and_save_batches(valid_batches, valid_mod_path, steps=9)
augment_and_save_batches(test_batches, test_mod_path, steps=6)


# In[40]:


# creating the mod batches
# i downsized these resolutions for training time purposes.
# playing with the resolution isn't something i had much time to experiment with but could be useful.
train_mod_batches = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0
).flow_from_directory(
    train_mod_path,
    target_size=(224, 224),
    batch_size=6,
    class_mode='categorical'
)

valid_mod_batches = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0
).flow_from_directory(
    valid_mod_path,
    target_size=(224, 224),
    batch_size=96,
    class_mode='categorical'
)
test_mod_batches = ImageDataGenerator(
    preprocessing_function=lambda x: x / 255.0
).flow_from_directory(
    test_mod_path, 
    target_size = (224,224), 
    batch_size = 74, 
    class_mode = 'categorical'
    )


# In[41]:


# building the cnn
# I started very simple a while back and gradually added all of these layers. it showed no significant improvement from the basic model.
# it does better than guessing but it usually is around high 50% to low 60% for accuracy (guessing is 33%)
# i pivoted to the vit which generally performs better (getting low 70% range usually) but it may still be worth playing around with this architecture.
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), data_format='channels_last'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation = 'softmax'),
])


# In[42]:


# compiling the cnn model
model.compile(Adam(learning_rate=.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[43]:


# takes python generator and makes a tensorflow dataset object from it
# (cnn and vit can be picky about which one is used so i left both in because i was frequently changing approaches that needed one or the other)
def generator_to_tf_dataset(generator, batch_size):
    output_signature = (
        tf.TensorSpec(shape=(batch_size, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, 3), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=output_signature
    )
    dataset = dataset.repeat()
    return dataset


# In[44]:


# create the tensorflow datasets
train_tf_dataset = generator_to_tf_dataset(train_mod_batches, batch_size=6)
valid_tf_dataset = generator_to_tf_dataset(valid_mod_batches, batch_size=96)
test_tf_dataset = generator_to_tf_dataset(test_mod_batches, batch_size=74)


# In[45]:


# calculates the proper steps per epoch based on the batch size
train_steps = train_mod_batches.n // train_mod_batches.batch_size
valid_steps = valid_mod_batches.n // valid_mod_batches.batch_size

# train the model
history = model.fit(
    x=train_tf_dataset,
    steps_per_epoch=train_steps,
    validation_data=valid_tf_dataset,
    validation_steps=valid_steps,
    epochs=20,
    verbose=2
)


# In[46]:


# plotting accuracy in the history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# plotting loss in the history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[47]:


# lists for predictions and true labels
all_predictions = []
all_true_labels = []

# make predictions for every image in each test batch
for batch_images, batch_labels in test_tf_dataset:
    # apply model to predict test images
    predictions = model.predict(batch_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(batch_labels, axis=1)

    all_predictions.extend(predicted_classes)
    all_true_labels.extend(true_classes)

    if len(all_true_labels) >= test_mod_batches.n:
        break

# make the confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)
class_names = list(test_mod_batches.class_indices.keys())

# plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# classification report
report = classification_report(all_true_labels, all_predictions, target_names=class_names)
print("Classification Report:")
print(report)

