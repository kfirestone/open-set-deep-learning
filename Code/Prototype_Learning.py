import cv2
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB0
from tqdm.keras import TqdmCallback

# ---------CONFIG---------
IMAGE_SIZE = 224
IMAGE_DIR = '../openset-split/split_data_set'
NUM_CLASSES = 15
EPOCHS = 100
BATCH_SIZE = 32

#Function to load in data from image set splits into proper size and format to feed into model
def load_data(dir, image_size, data_set):
    '''
    Loads in images, resizes images and stores in NumPy array along with labels

    Arguments:
    dir: Base directory of images. Subfolders should be "known_classes" and "unknown_classes"
    image_size: image size to be converted to. Images will be returned as size image size x image size
    data_set: Which data set to load. Must to either "known" or "unknown" anything else will throw an exception

    Returns:
    If data_set is "known" returns 3 tuples of the form (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    If data_set is "unknown" return 2 tuples of the form (x_valid, y_valid), (x_test, y_test)

    The y arrays are NOT one hot encoded.
    '''

    x_train = []
    y_train = []

    x_valid = []
    y_valid = []

    x_test = []
    y_test = []

    if data_set not in ['known', 'unknown']:
        print('Invalid data set. Expected "known" or "unknown".')
        return

    class_list = os.listdir(dir + '/' + data_set + '_classes/valid/')

    for i, cls in enumerate(tqdm(class_list)):

        if data_set == 'known':
            for image in os.listdir(dir + '/' + data_set + '_classes/train/' + cls + '/'):
                img = cv2.imread(dir + '/' + data_set + '_classes/train/' + cls + '/' + image)
                if img is None:
                    pass
                else:
                    img = cv2.resize(img, (image_size, image_size))
                    x_train.append(img)
                    y_train.append(i)

        for image in os.listdir(dir + '/' + data_set + '_classes/valid/' + cls + '/'):
            img = cv2.imread(dir + '/' + data_set + '_classes/valid/' + cls + '/' + image)
            if img is None:
                pass
            else:
                img = cv2.resize(img, (image_size, image_size))
                x_valid.append(img)
                y_valid.append(i)

        for image in os.listdir(dir + '/' + data_set + '_classes/test/' + cls + '/'):
            img = cv2.imread(dir + '/' + data_set + '_classes/test/' + cls + '/' + image)
            if img is None:
                pass
            else:
                img = cv2.resize(img, (image_size, image_size))
                x_test.append(img)
                y_test.append(i)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    if data_set == 'known':
        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
    else:
        return (x_valid, y_valid), (x_test, y_test)
    

#Create custom layer for model prototypes
class Prototype_Layer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.prototype = self.add_weight(shape = (15, 15), initializer = keras.initializers.Zeros(), trainable = True)
        
    def call(self):
        return self.prototype


#Create custom layer for model prototype radii
class Prototype_Radius(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.prototype_radii = self.add_weight(shape = (1, 15), initializer = keras.initializers.Zeros(), constraint = keras.constraints.NonNeg(), trainable = True)
        
    def call(self):
        return self.prototype_radii


#Create custom model incorporating prototype layer, prototype radii layer and custom loss functions
class Prototype_Model(keras.Model):
    def __init__(self):
        super().__init__()
        self.EffNet = EfficientNetB0(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3), classes = NUM_CLASSES)
        self.pooling = keras.layers.GlobalAveragePooling2D()
        self.batch_norm = keras.layers.BatchNormalization()
        self.dropout = keras.layers.Dropout(0.2)
        self.features = keras.layers.Dense(15)
        self.softmax = keras.layers.Softmax()
        self.prototype_layer = Prototype_Layer().prototype
        self.prototype_radii = Prototype_Radius().prototype_radii
        
    def call(self, inputs):
        images, labels_OHE = inputs
        x = self.EffNet(images)
        x = self.pooling(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        image_features = self.features(x)
        preds = self.softmax(image_features)
        
        pred_labels = tf.argmax(preds, axis = 1)
        pred_labels = tf.reshape(pred_labels, (-1, 1))
        
        labels = tf.argmax(labels_OHE, axis = 1)
        labels = tf.reshape(labels, (-1, 1))
        
        correct_preds = pred_labels[pred_labels == labels]
        correct_preds = tf.reshape(correct_preds, (-1, 1))
        
        image_features = tf.reshape(image_features, (-1, 15))
        
        pred_prototypes = tf.gather_nd(indices = labels, params = self.prototype_layer)
        
        D = 1/(tf.math.reduce_sum(tf.math.square(tf.expand_dims(image_features, axis = 1) - self.prototype_layer), axis = 2) + 0.001)
        
        d_t = tf.math.reduce_max(tf.gather_nd(indices = correct_preds, params = D), axis = 1)
        r_t = tf.gather_nd(indices = tf.concat([tf.zeros((correct_preds.shape[0], 1), dtype = tf.int64), correct_preds], axis = 1), params = self.prototype_radii)
        
        l2_prototype_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(image_features - pred_prototypes), axis = 1))
        distance_loss = -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(tf.cast(labels_OHE, tf.float32), tf.math.log(tf.nn.softmax(D, axis = 1))), axis = 1))
        
        radius_loss = tf.math.reduce_mean(tf.math.square(r_t - d_t))
        
        prototype_loss = 0.1*(l2_prototype_loss + distance_loss) + 0.01*(radius_loss)
        
        if tf.math.is_nan(prototype_loss):
            prototype_loss = 0
        
        self.add_loss(prototype_loss)
        
        return preds, pred_prototypes

if __name__ == '__main__':

    #Load data & format
    (x_train, y_train), (x_valid,  y_valid), (x_test, y_test) = load_data(IMAGE_DIR, IMAGE_SIZE, 'known')

    y_train = keras.utils.to_categorical(y_train)
    y_valid = keras.utils.to_categorical(y_valid)
    y_test = keras.utils.to_categorical(y_test)

    #Initialize model
    prototype_model = Prototype_Model()

    #Set training parameters
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    classification_loss = keras.losses.CategoricalCrossentropy()
    metric = keras.metrics.CategoricalAccuracy()

    #Create data set to feed into model
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(BATCH_SIZE)

    #Model training loop
    running_total_loss = []
    running_classification_loss = []
    running_classification_accuracy = []
    running_prototype_loss = []
    running_prototypes = []
    running_radii = []

    for epoch in tqdm(range(EPOCHS)):
        batch_total_loss = 0
        batch_classification_loss = 0
        batch_correct_classifications = 0
        batch_prototype_loss = 0
        
        for inputs in train_dataset:
            _, labels = inputs
            
            with tf.GradientTape() as tape:
                preds, prototypes = prototype_model(inputs)
                loss = classification_loss(labels, preds) + sum(prototype_model.losses)
                
            batch_total_loss += loss
            batch_classification_loss += classification_loss(labels, preds)
            batch_correct_classifications += (tf.argmax(preds, axis = 1) == tf.argmax(labels, axis = 1)).numpy().sum()
            batch_prototype_loss += sum(prototype_model.losses)
            
            grads = tape.gradient(loss, prototype_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, prototype_model.trainable_weights))
            
        running_total_loss.append(batch_total_loss/x_train.shape[0])
        running_classification_loss.append(batch_classification_loss/x_train.shape[0])
        running_classification_accuracy.append(batch_correct_classifications/x_train.shape[0])
        running_prototype_loss.append(batch_prototype_loss/x_train.shape[0])
        running_prototypes.append(prototype_model.prototype_layer)
        running_radii.append(prototype_model.prototype_radii)
        
    running_total_loss = np.array(running_total_loss)
    running_classification_loss = np.array(running_classification_loss)
    running_classification_accuracy = 100*np.array(running_classification_accuracy)
    running_prototype_loss = np.array(running_prototype_loss)
    running_prototypes = np.array(running_prototypes)
    running_radii = np.array(running_radii)

    #Save training statistics and model weights
    prototype_model.save_weights('./prototype_model_weights/EffNetd0_prototype_model')
    np.save('./running_total_loss', running_total_loss)
    np.save('./running_classification_loss', running_classification_loss)
    np.save('./running_classification_accuracy', running_classification_accuracy)
    np.save('./running_prototype_loss', running_prototype_loss)
    np.save('./running_prototypes', running_prototypes)
    np.save('./running_radii', running_radii)