import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
import itertools
import sklearn.metrics as metrics
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
#Need to edit below import if using base model other than DenseNet
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam, SGD
#Need to edit below import if using base model other than DenseNet
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

#Setting seeds so that randomization is kept consistent.
np.random.seed(70) 
random.seed(70) 
tf.random.set_seed(70)

#CSV logger automatically keeps track of accuracy and loss for both training and validation during each epoch.
#This is very convenient for graphing model performance through time later in R
#Outputs a .csv file
log_csv = CSVLogger('my_logs_robocrop.csv', separator=',', append = False)
callbacks_list = [log_csv]

#This is the dimensions that each image will be shaped to
#DenseNet201 requires 224 x 224.
img_height, img_width = (224,224)
#Batch size is the number of training examples utilized in one iteration. Could use 16, may increase compute time.
batch_size = 32

#These are the directories/folders where your images are stored for training, validation, and test datasets.
train_data_dir = r"full_images_2021/train_full"
valid_data_dir = r"full_images_2021/validation_full"
test_data_dir = r"full_images_2021/test_full"

#ImageDataGenerator performs image augmentation for each image on the fly. Rotating, flipping, brightness, etc.
train_datagen = ImageDataGenerator(
    width_shift_range= 0.2, # 0.2 fraction of total width/height
    height_shift_range=0.2,
    fill_mode="nearest",
    brightness_range=[0.9,1.1], #range for picking a shift value from
    rotation_range =30, #degree range for random rotations
    vertical_flip = True,
    horizontal_flip = True,
    validation_split = 0.05,
    rescale=1./255) #rescaling image pixel values by the number of channels, 1/255

#Pulls your training dataset images
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical', #more than 2 classes --> categorical
    subset='training') #set as training data

#Pulls your validation dataset images
valid_generator = train_datagen.flow_from_directory(
    valid_data_dir, #same directory as training data
    target_size = (img_height, img_width),
    batch_size=batch_size,
    class_mode= 'categorical',
    subset='validation') #set as validation data

#Pulls your test dataset images. Note that you only want to use 1 image at a time for test.
test_generator = train_datagen.flow_from_directory(
    test_data_dir, #same directory as training data
    target_size = (img_height, img_width),
    batch_size=1,
    class_mode= 'categorical',
    subset='validation') #set as validation data

#Loads array and classes of items in test folder for each batch
x,y = test_generator.next()
x.shape

#Sets base model architecture, could swap out DenseNet201 for ResNet50, VGG16, etc
# Whether to include the fully-connected layer at top of network
# 'imagenet' (pre-training on ImageNet), or the path to the weights file to be loaded.
base_model = DenseNet201(include_top=False, weights='imagenet')
#Below x lines are the additional architecture attached beyond the base model
x = base_model.output
#Globalaveragepooling performs the 'flatten' purposes
#x = Flatten()x
x = GlobalAveragePooling2D()(x)
#128, 64 neuron fully-connected layers with relu activation
#Dropout for regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
#Predicting a label for each image based on softmax activation, for 8 classes
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

#Deciding whether to freeze the base model weights (imagenet) or allow them to update
#Setting this to 'True' implies we are training the entire model
for layer in base_model.layers:
    layer.trainable = True

#Compile the model using stochastic gradient descent w/ learning rate and momentum values
#Use categorical crossentropy as the loss function
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
#Train the model for 100 epochs, verbose just tells you much info to display during training
model.fit(train_generator, epochs = 100, verbose=1, validation_data = valid_generator, callbacks=callbacks_list)

#Add the below block if you want to experiment with freezing the base weights for the above epochs.
# Note, would need to change above to layer.trainable=False to freeze the above epochs.
# Would likely change the learning rate above too.
# And then unfreeze them for additional epochs below. (layer.trainable=True) 
"""
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs = 10, verbose=1, validation_data = valid_generator, callbacks=callbacks_list)"""

#The big block below lays out the steps for calculating and plotting
# a classification report (precision, recall, etc.) and confusion matrix 
# I prefer to plot manually using wcipriano's pretty print confusion matrix
"""
test_steps_per_epoch = np.math.ceil(valid_generator.samples / valid_generator.batch_size)
Y_pred = model.predict_generator(valid_generator, steps= test_steps_per_epoch)
true_classes = valid_generator.classes
predicted_classes = np.argmax(Y_pred, axis=1)
class_labels = list(valid_generator.class_indices.keys())
cm1 = confusion_matrix(true_classes, predicted_classes)
print(cm1)

print('Classification Report')
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
   
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Compute Confusion Matrix
cnf_matrix = confusion_matrix(true_classes, predicted_classes)
cm_plot_labels = ['bacterial_blight', 'cercospora_leaf_blight','downey_mildew', 'frogeye', 'non_disease', 'potassium_deficiency', 'soybean_rust', 'target_spot']
np.set_printoptions(precision=2)
#Plot non-normalized cm
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=cm_plot_labels, title='Confusion Matrix, Without Normalization')
plt.savefig("without_normalized.png")

#Plot normalized cm
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=cm_plot_labels, normalize=True, title='Normalized Confusion Matrix')
plt.savefig("normalized.png")
"""

#Evaluate the performance of the model using test_generator
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest Accuracy:', test_acc)

#Save the model and weights
#Mainly need the .h5 file to best hosted on the webapp
model.save_weights('saved_model/dense_weights_1118')
model.save('saved_model/dense_1118.h5')
model.save('saved_model/dense_1118.hdf5')
