from google.colab import drive
drive.mount('/content/drive')

!pip install -U --pre efficientnet

from efficientnet.keras import EfficientNetB7

import numpy as np
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import decode_predictions
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile
from keras.models import load_model

pre_trained_model=EfficientNetB7(weights='imagenet',include_top=False)  
pre_trained_model.summary()

local_zip = '//content/drive/My Drive/Firedetectorfinal.zip'

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/Firedetectorfinal'

train_dir = os.path.join( base_dir, 'training')
validation_dir = os.path.join( base_dir, 'validation')


train_fire_dir = os.path.join(train_dir, 'Fire') # Directory with our training fire pictures
train_nonfire_dir = os.path.join(train_dir, 'NonFire') # Directory with our training nonfire pictures
validation_fire_dir = os.path.join(validation_dir, 'Fire') # Directory with our validation fire pictures
validation_nonfire_dir = os.path.join(validation_dir, 'NonFire')# Directory with our validation nonfire pictures

train_fire_fnames = os.listdir(train_fire_dir)
train_nonfire_fnames = os.listdir(train_nonfire_dir)

train_datagen = ImageDataGenerator( rescale = 1./255. )
val_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 32,
                                                    class_mode = 'binary', 
                                                    target_size = (100,100))     

validation_generator =  val_datagen.flow_from_directory( validation_dir,
                                                          batch_size  = 24,
                                                          class_mode  = 'binary', 
                                                          target_size = (100, 100))

last_layer =pre_trained_model.get_layer('block7d_se_reshape')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


for layer in pre_trained_model.layers:
  layer.trainable = True

from keras import layers
from keras import Model

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

history = model.fit_generator(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 656,
            epochs = 10,
            validation_steps = 100,
            verbose = 1)

model.save('fnet.h5')