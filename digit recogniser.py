#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from subprocess import check_output


# In[3]:


train=pd.read_csv("digit train.csv")
test=pd.read_csv("digit test.csv")


# In[4]:


test.head()



# In[5]:


y_train=train["label"].astype("float32")
x_train=train.drop(["label"],axis=1).astype("int32")
x_test=test.astype("float32")
x_train.shape,y_train.shape


# In[6]:


x_train=x_train.values.reshape(-1,28,28,1)
x_train=x_train/255.0
x_test=x_test.values.reshape(-1,28,28,1)
x_test=x_test/255.0
x_test.shape,x_train.shape


# In[7]:


y_train=tf.keras.utils.to_categorical(y_train,10)


# In[8]:


train["label"].head()


# In[9]:


print(y_train[0:9, :])


# In[10]:


model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    tf.keras.layers.Dropout(0.25),  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.Dense(256,activation="relu"),                        
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(10,activation="softmax")
                           ])
                           
model.summary()                           
                            


# In[11]:



class callback(tf.keras.callbacks.Callback):
    def epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>0.999):
            print("reached 99.5% accuracy")
            self.module.stop_training=True
callbacks=callback()            


# In[11]:


optimizer=tf.keras.optimizers.Adam(
     learning_rate=0.0005,
     beta_1=0.9,
     beta_2=0.999,
     epsilon=1e-07,
     name="Adam"

)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size=50, epochs=20, callbacks=[callbacks])


# In[12]:


pred=model.predict(x_test)
pred=np.argmax(pred,axis=1)
pred=pd.Series(pred,name="Label")
out= pd.concat([pd.Series(range(1,28001),name="ImageId"),pred],axis=1)
out.to_csv('submission 2.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




