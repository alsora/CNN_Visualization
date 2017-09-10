import os
os.environ['GLOG_minloglevel'] = '3'

import caffe
import numpy as np
import sys
import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def backspace(n):
    sys.stdout.write('\r'+n)
    sys.stdout.flush()

class CaffeNet():
    """The class initializes a caffenet and brings some usefull methods.

    Parameters
    ---------
    model_path: the path to the prototxt
    weights_path: the path to the weights
    mean_path: the path to the mean of the dataset, default: None
    image_scale: the color scale accepted by the net, default: 255.0
    batch_size: batch size for the forward pass, default: 1
    input_shape: (rows, cols) the input shape of the net, default: (227, 227)"""

    def __init__(self, model_path,
                 weights_path,
                 mean_path=None,
                 image_scale=255.0,
                 batch_size=1,
                 input_shape=(227, 227)):

        self.net = caffe.Net(model_path,      # defines the structure of the model
                        weights_path,  # contains the trained weights
                        caffe.TEST)     # use test mode (e.g., don't perform dropout)

        self.net.blobs['data'].reshape(batch_size, 3, input_shape[0], input_shape[1])
        self.net.blobs['prob'].reshape(batch_size, )

        self.mean_path = mean_path
        self.image_scale = image_scale
        self.batch_size = batch_size

        self.transformer = self.set_transformer()

    def set_transformer(self):
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1)) #move image channels to outermost dimension
        transformer.set_channel_swap('data', (2, 1, 0)) # if using RGB instead of BGR
        transformer.set_raw_scale('data', self.image_scale)
        
        if self.mean_path:
            transformer.set_mean('data', np.load(self.mean_path).mean(1).mean(1))

        return transformer

    def preprocess_images(self, image_set):
        transformed_images = []

        for image in image_set:
            transformed_images.append(self.transformer.preprocess('data', image))

        return transformed_images

    def deprocess_images(self, image_set):
        transformed_images = []

        for image in image_set:
            transformed_images.append(self.transformer.deprocess('data', image))

        return transformed_images

    def get_features(self, batch, extraction_layer, most_active_filter=None):

        features_vector = []
        
        for image in batch:

            self.net.blobs['data'].data[...] = image
            self.net.forward()
            
            features = self.net.blobs[extraction_layer].data[0]
            if most_active_filter is int:
                features_vector.append(features[most_active_filter].copy())
            else:
                features_vector.append(features.copy())

        return features_vector

    @staticmethod
    def get_most_active_filters(images_features, n=10):

        best_filters = []

        for filters in images_features:

            mean_filters = [np.mean(filter_) for filter_ in filters]

            # reversing the order
            filters_sorted = np.argsort(mean_filters)[::-1]
            best_filters.append(filters_sorted[:n])

        return best_filters

    def get_probs(self, batch):

        batch_probabilities = []
        for img in batch:
            self.net.blobs['data'].data[...] = img
            batch_output = self.net.forward()

            batch_probabilities.append(batch_output['prob'][0])

        return batch_probabilities

    def get_probs_and_features(self, batch, extraction_layer, most_active_filter=None):

        batch_probabilities = []
        features_vector = []

        for img in batch:
            
            self.net.blobs['data'].data[...] = img
            batch_output = self.net.forward()
            batch_probabilities.append(batch_output['prob'][0].copy())
            
            features = self.net.blobs[extraction_layer].data[0]
            if most_active_filter is int:
                features_vector.append(features[most_active_filter].copy())
            else:
                features_vector.append(features.copy())
            
        return batch_probabilities, features_vector

    @staticmethod
    def batch_iterator(images, batch_size):
        batch = []
        for idx, image in enumerate(images):
            batch.append(image)

            idx+=1

            if idx % batch_size == 0 and idx != 0:
                yield batch
                batch = []


    @staticmethod
    def get_precision(trueLabels, predictedLabels):

        countCorrect1 = 0
        countCorrect5 = 0

        if len(trueLabels) != len(predictedLabels):
            print 'True and Predicted lists have different size.'
            print len(trueLabels), " True labels"
            print len(predictedLabels), " Predicted labels"

            return 0

        for index, item in enumerate(trueLabels):
            
            prediction = predictedLabels[index]
            if prediction[0] == item:
                countCorrect1 += 1

            if item in prediction:
                countCorrect5 += 1

        percentage1 = 100.0 * countCorrect1/len(trueLabels)
        percentage5 = 100.0 * countCorrect5/len(trueLabels)

        print percentage1, ' % Top1 Correct predictions'
        print percentage5, ' % Top5 Correct predictions'


        return percentage1


    @staticmethod
    def plot_confusion_matrix(truePredicted, inlierPredicted, classes,
                              title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        cm = confusion_matrix(truePredicted, inlierPredicted)

        cmap=plt.cm.Blues

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        np.set_printoptions(precision=2)

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, np.around(cm[i, j], decimals=2),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig(title + ".png")

        return

    @staticmethod
    def outputs_to_synsets(output, word_synsets):
        synsets = []

        for value in output:
            synset = word_synsets[value]
            synsets.append(synset)

        return synsets

    @staticmethod
    def synsets_to_words(synsets):
        new_labels = []

        for synset in synsets:

            word = synset.split(',')[0]
            new_labels.append(word)

        return new_labels
