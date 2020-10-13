#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
os.getcwd()


# In[2]:


# re-size all the images to this
IMAGE_SIZE = [400, 400]
train_path = '../data_prep/data_for_modelling/train'
valid_path = '../data_prep/data_for_modelling/val'


# In[3]:


# add preprocessing layer to the front of VGG
base_model = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in base_model.layers:
    layer.trainable = False
base_model.summary()


# In[4]:


train_folders = glob(os.path.join(train_path, '*'))
train_folders


# In[5]:


x = Flatten()(base_model.output)
prediction = Dense(len(train_folders), activation='softmax')(x)


# In[6]:


model = Model(inputs=base_model.input, outputs=prediction)
model.summary()


# In[7]:


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[8]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./255)


# In[10]:


training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

val_set = val_datagen.flow_from_directory(valid_path,
                                            target_size = IMAGE_SIZE,
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[ ]:


r = model.fit(
  training_set,
  validation_data=val_set,
  epochs=2,
  steps_per_epoch=len(training_set),
  validation_steps=len(val_set)
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




