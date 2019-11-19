import numpy as np
import os
from keras.models import load_model
from efficientnet.keras import EfficientNetB7
import cv2
from keras.applications.imagenet_utils import decode_predictions
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import zipfile
from tqdm import tqdm

local_zip = '/content/drive/My Drive/sharmads.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/sharmads'
DATADIR = base_dir
CATEGORIES = ['Fire', 'NonFire']

IMG_SIZE = 100

X = []
Y = []
for category in CATEGORIES:  
    path = os.path.join(DATADIR,category) 
    class_num = CATEGORIES.index(category)  
        
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img))  
        if os.path.join(path,img)=="/tmp/sharmads/Fire/.DS_Store":
          continue
        if os.path.join(path,img)=="/tmp/sharmads/NonFire/.DS_Store":
          continue  
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        X.append(new_array)
        Y.append(class_num)
        
os.environ['KMP_DUPLICATE_LIB_OK']='True'

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = X/255.0
X.shape[1:]

model=load_model('fnet.h5')
loss, accuracy= model.evaluate(X,Y, verbose=1)
print(model.metrics_names)
print(loss, accuracy)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()
