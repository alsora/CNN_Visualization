import caffe
import os
import numpy as np
import sys
import itertools

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


os.environ['GLOG_minloglevel'] = '3'


def backspace(n):
    sys.stdout.write('\r'+n)
    sys.stdout.flush()


def preprocessImages(imageSet, net, meanPath='', imageScale = 255.0):

    transformedImages = []

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1)) #move image channels to outermost dimension
    transformer.set_channel_swap('data', (2, 1, 0)) # if using RGB instead of BGR
    transformer.set_raw_scale('data', imageScale)
    if meanPath is not '':
        transformer.set_mean('data', np.load(meanPath).mean(1).mean(1))

    for image in imageSet:
        transformedImages.append(transformer.preprocess('data', image))

    return transformedImages
    

def extractFeatures(imageSet, net, extractionLayerName):

    featuresVector = []
    totalImages = len(imageSet)
    for num, image in enumerate(imageSet):

        net.blobs['data'].data[...] = image
        net.forward()
        
        features = net.blobs[extractionLayerName].data[0]
        
        featuresVector.append(features.copy().flatten())
        
        string_to_print = '{} of {}'.format(num + 1, totalImages)
        backspace(string_to_print)

    print '\n'

    return featuresVector


def predictLabels(imageSet, net):

    labelsVector = []
    totalImages = len(imageSet)
    for num, image in enumerate(imageSet):

        net.blobs['data'].data[...] = image
        output = net.forward()

        #plabel = int(output['prob'].argmax())
        best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-5:-1]

        labelsVector.append(best_n.copy())
        
        string_to_print = '{} of {}'.format(num + 1, totalImages)
        backspace(string_to_print)

    print '\n'

    return labelsVector


def batch_iterator(images, batch_size):
    batch = []
    for idx, image in enumerate(images):
        batch.append(image)

        idx+=1

        if idx % batch_size == 0 and idx != 0:
            yield batch
            batch = []

def occlusion_features_and_labels(batch, net, extraction_layer_name):

    batch_probabilities = []
    for img in batch:
        net.blobs['data'].data[...] = img
        batch_output = net.forward()

        #batch_features = net.blobs[extraction_layer_name].data
        #batch_labels = batch_output['prob']
        batch_probabilities.append(batch_output['prob'][0])
        
    return batch_probabilities

def getFeaturesAndLables(imageSet, net, extractionLayerName):

    featuresVector = []
    labelsVector = []
    totalImages = len(imageSet)
    for num, image in enumerate(imageSet):

        net.blobs['data'].data[...] = image
        output = net.forward()
        
        features = net.blobs[extractionLayerName].data[0]
        #plabel = int(output['prob'].argmax())
        best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-5:-1]
        featuresVector.append(features.copy().flatten())
        #labelsVector.append(plabel)
        labelsVector.append(best_n.copy())

        string_to_print = 'Images processed: {} of {}'.format(num + 1, totalImages)
        backspace(string_to_print)

    print '\n'

    return featuresVector, labelsVector


def getPrecision(trueLabels, predictedLabels):

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


def outputToSynsets(output, word_synsets):

    synsets = []

    for value in output:
        synset = word_synsets[value]
        synsets.append(synset)

    return synsets


def synsetsToWords(synsets, synset_words):

    newLabels = []

    for synset in synsets:

        index = np.where(synset_words[:,0] == synset)

        fullDescription = synset_words[index[0],1][0]

        description = fullDescription.split(',')[0]

        newLabels.append(description)

    return newLabels
