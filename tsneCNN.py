import os
os.environ['GLOG_minloglevel'] = '3'

import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from sklearn.metrics import confusion_matrix

from dataset_utils import loadImageNetFiles

import tsne

import caffeCNN_utils as CNN


netLayers = {
    'caffenet': 'fc7'
}


def main(argv):

    pycaffe_path = os.path.dirname(caffe.__file__)
    caffe_path = os.path.normpath(os.path.join(pycaffe_path, '../../'))
    mean_path = os.path.join(pycaffe_path,'imagenet/ilsvrc_2012_mean.npy')
    synsets_path = os.path.join(caffe_path,'data/ilsvrc12/synsets.txt')



    model_filename = os.path.join(caffe_path, 'models/bvlc_reference_caffenet/deploy.prototxt')
    weight_filename = os.path.join(caffe_path, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    cnn_type = 'caffenet'
    images_dir = ''
    caffe.set_mode_cpu()

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="caffemodel file")
    parser.add_argument("-p", "--prototxt", help="prototxt file")
    parser.add_argument("-i", "--input_im", help="input images dir")
    parser.add_argument("-n", "--net_type", help="cnn type (resnet/googlenet/vggnet")
    parser.add_argument("-g", "--gpu", help="enable gpu mode", action='store_true')
    args = parser.parse_args()

    if args.prototxt:
        model_filename = args.prototxt
    if args.weights:
        weight_filename = args.weights
    if args.input_im:
        images_dir = args.input_im
    if args.net_type:
        cnn_type = args.net_type
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(0)


    if os.path.isfile(model_filename):
        print 'Caffe model found.'
    else:
        print 'Caffe model NOT found...'
        sys.exit(2)


    extractionLayerName = netLayers[cnn_type]
    synset_words = np.loadtxt(synsets_path, str, delimiter='\t')

    net = caffe.Net(model_filename,      # defines the structure of the model
                    weight_filename,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)


    net.blobs['data'].reshape(1,3,227,227)



    #Create Images and labels
    numberImagesPerClass = 50

    classes = ['n02119789', 'n03773504', 'n04254680', 'n04429376', 'n04507155']
    #If you don't provide synsets, it will take images from all the sub-folders in images_dir
    
    names, images, labels = loadImageNetFiles(images_dir, numberImagesPerClass, classes)
    #Preprocess images

    images = CNN.preprocessImages(images, net, mean_path)

    #Forward pass to extract features	

    features, predictedLabels = CNN.getFeaturesAndLables(images, net, extractionLayerName)

    predictedLabels = CNN.outputToSynsets(predictedLabels, synset_words)

    CNN.getPrecision(labels, predictedLabels)

    #Apply tsne

    perplexity = 10.0
    tolerance = 1e-5
    iterations = 800

    features = np.array(features)

    P = tsne.computeProbabilities(features, perplexity, tolerance)
    positions = tsne.computeMapPoints(P, iterations)

    #map labels to something more meaningful

    classes = set(labels)  #get unique values

    classesToNumbers = dict()

    for index, item in enumerate(classes):
        classesToNumbers[item] = index

    mappedLabels = [classesToNumbers.get(key) for key in labels]
    mappedPredictedLabels = [classesToNumbers.setdefault(key, len(classes) + 1) for key in predictedLabels]

    tsne.showPoints(positions, mappedLabels)
    tsne.showPoints(positions, mappedPredictedLabels)











if __name__=='__main__':
    main(sys.argv[1:])
