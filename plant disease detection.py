#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os,os.path
import splitfolders
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[7]:


matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
DATA_DIR = '/kaggle/input/plantvillage-dataset/color'
BATCH_SIZE = 64
EPOCHS = 5
IMAGE_SHAPE = (224, 224)


# In[8]:


def f1_macro(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[4]:


pairs = list()
number = list()

for directory in os.listdir(path=DATA_DIR):
    columns = directory.split('___')
    columns.append(directory)
    
    sub_path = DATA_DIR + '/' + directory
    columns.append(len([name for name in os.listdir(path=sub_path)]))
    
    pairs.append(columns)


# In[ ]:


pairs = pd.DataFrame(pairs, columns=['Plant', 'Disease', 'Directory', 'Files'])
pairs.sort_values(by='Plant')


# In[ ]:


rows_to_drop = [17, 3, 33, 4, 5]
dir_to_delete = pairs[pairs.index.isin(rows_to_drop)]['Directory']
print(dir_to_delete.values)
['Orange___Haunglongbing_(Citrus_greening)' 'Soybean___healthy'
 'Squash___Powdery_mildew' 'Blueberry___healthy' 'Raspberry___healthy']


# In[ ]:


os.mkdir('img')
os.mkdir(os.path.join('img', 'train'))
os.mkdir(os.path.join('img', 'val'))
os.mkdir(os.path.join('img', 'test'))


# In[ ]:


splitfolders.ratio(DATA_DIR,output = "img",seed = 42,ratio = (0.80,0.10,0.10))


# In[ ]:


TRAIN_PATH = "./images/train"
VAL_PATH = "./images/val"
TEST_PATH  = "./images/test"
PATHS = [TRAIN_PATH, VAL_PATH, TEST_PATH]


# In[ ]:



for sub_directory in dir_to_delete.values:
    for directory in PATHS:
        d = directory + '/' + sub_directory
        shutil.rmtree(d)


# In[ ]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_gen = datagen.flow_from_directory(directory = TRAIN_PATH, 
                                          class_mode="categorical",
                                          target_size = IMAGE_SHAPE,
                                          batch_size = BATCH_SIZE,
                                          color_mode='rgb',
                                          seed = 1234,
                                          shuffle = True)

val_gen = datagen.flow_from_directory(directory = VAL_PATH, 
                                          class_mode="categorical",
                                          target_size = IMAGE_SHAPE,
                                          batch_size = BATCH_SIZE,
                                          color_mode='rgb',
                                          seed = 1234,
                                          shuffle = True)

test_gen = datagen.flow_from_directory(directory = TEST_PATH, 
                                          class_mode="categorical",
                                          target_size = IMAGE_SHAPE,
                                          batch_size = BATCH_SIZE,
                                          color_mode='rgb',
                                          shuffle = False)


# In[5]:


os.mkdir('models')
os.mkdir(os.path.join('models', 'first_version'))


# In[6]:


CHECKPOINT_PATH_MODEL_FIRST = "./models/first_version"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH_MODEL_FIRST,
                                      monitor='val_loss',
                                      save_best_only=True)


# In[22]:


early_stopping = EarlyStopping(monitor='val_loss', patience = 2, restore_best_weights=True)

vgg16_base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='max')

vgg16_base_model.trainable = False
inputs = vgg16_base_model.input


# In[23]:



x = BatchNormalization()(vgg16_base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.45, seed=1234)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.45, seed=1235)(x)
x = Flatten()(x)

outputs = Dense(33, activation='softmax')(x)


# In[24]:


outputs = Dense(33, activation='softmax')(x)

vgg16_model = Model(inputs=inputs, outputs=outputs)
vgg16_model.compile(
    optimizer=Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=[f1_macro]
)


# In[25]:


history = vgg16_model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=[checkpoint_callback, early_stopping])


# In[26]:


plt.plot(Epochs, history.history['loss'], label = 'training loss')
plt.plot(Epochs, history.history['val_loss'], label = 'validation loss')
plt.grid(True)
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.show()


# In[27]:


Epochs = [i+1 for i in range(len(history.history['f1_macro']))]

plt.plot(Epochs, history.history['f1_macro'], label = 'training f1 score')
plt.plot(Epochs, history.history['val_f1_macro'], label = 'validation f1 score')
plt.grid(True)
plt.legend()
plt.title('Training and Validation F1-score')
plt.xlabel('Epochs')
plt.show()


# In[9]:


import matplotlib.pyplot as plt
from PIL import Image

# Load the input image
img_path = 'plant.jpg'
img = Image.open(img_path)

# Display the input image
plt.imshow(img)
plt.axis('off')
plt.savefig('input_image.png')  # Save the input image as 'input_image.png'
plt.show()

# Predicted labels and probabilities
labels = ['Aconitum', 'viaduct', 'bell_cote']
probabilities = [40.37, 15.87, 5.66]  # Corresponding probabilities

# Display the bar chart
plt.bar(labels, probabilities)
plt.xlabel('Labels')
plt.ylabel('Probabilities')
plt.title('Top 3 Predictions')
plt.savefig('predictions_chart.png')  # Save the bar chart as 'predictions_chart.png'
plt.show()


# In[ ]:




