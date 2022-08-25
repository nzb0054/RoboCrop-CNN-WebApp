import numpy as np
import os
import random
import tensorflow as tf
import ssl
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model, Input
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.backend import manual_variable_initialization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import itertools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow import profiler
from tensorflow import keras
from keras_flops import get_flops
tf.enable_eager_execution()

model = keras.models.load_model('model_file_path')

# # Calculate FLOPs, may need to change batch size
flops = get_flops(model, batch_size=32)
print(f"FLOPS: {flops / 10 ** 9:.03} G")


import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model

model = tf.keras.models.load_model('model_file_path')

t00=time.time()
graph = tf.Graph()
graph_def = tf.compat.v1.GraphDef()

with graph.as_default():
    tf.import_graph_def(graph_def)


t01=time.time()
print("Iniialization time",t01-t00)

folder_path = os.path.abspath('images_file_path')
images = []
for img in os.listdir(folder_path):
    cwd = os.getcwd()
    files = os.listdir(cwd)
    print("Files in %r: %s" % (cwd, files))
    img = image.load_img(img, color_mode='rgb', target_size=(224,224,3))
    img = tf.keras.preprocessing.image.array_to_img(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    # img = img.reshape((-1,) + img.shape)
    images.append(img)
images = np.vstack(images)
# classes = classifier.predict(images, batch_size=32)

print(img.dtype)
print(img.shape)


t0=time.time()

#Batch size has the model look at 32 images, so to get the average time for 1 image, divide the inference time by 32.
y_pred = model.predict(images, batch_size=32)
print(y_pred)
classname = y_pred[0]
print("Class: ",classname)

t1 = time.time()
print("Inference time", t1-t0)
