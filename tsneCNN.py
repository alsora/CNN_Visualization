import os
os.environ['GLOG_minloglevel'] = '3'

import warnings
warnings.filterwarnings("ignore")


import caffe
import sys
import argparse
import numpy as np

import tsne
#import caffeCNN_utils as CNN
import caffe_utils

from dataset_utils import loadImageNetFiles

netLayers = {
    'caffenet': 'fc7',

   #'googlenet': 'inception_5b/output'
   'googlenet': 'loss3/classifier',
   'resnet': 'fc1000'
}


def main(argv):

    pycaffe_path = os.path.dirname(caffe.__file__)
    caffe_path = os.path.normpath(os.path.join(pycaffe_path, '../../'))
    mean_path = os.path.join(pycaffe_path, 'imagenet/ilsvrc_2012_mean.npy')
    synsets_num_path = os.path.join(caffe_path, 'data/ilsvrc12/synsets.txt')
    synsets_to_class_path = os.path.join(os.getcwd(), 'synset_words.txt')

    model_filename = os.path.join(caffe_path, 'models/bvlc_reference_caffenet/deploy.prototxt')
    weight_filename = os.path.join(caffe_path, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    cnn_type = 'caffenet'
    images_dir = 'Images'
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

    # building dictionaries and inverse ones
    idx_to_synset = {}
    synset_to_idx = {}

    with open(synsets_num_path, 'r') as fp:
        for idx, synset in enumerate(fp):
            synset = synset.strip()
            idx_to_synset[idx] = synset
            synset_to_idx[synset] = idx

    synset_to_class = {}
    class_to_synset = {}

    with open(synsets_to_class_path, 'r') as fp:
        for line in fp:
            [synset, class_] = line.strip().split(' ', 1)
            synset_to_class[synset] = class_
            class_to_synset[class_] = synset

 	#Loading net and utilities    
    net = caffe_utils.CaffeNet(model_filename, weight_filename, mean_path)

    #Create Images and labels
    numberImagesPerClass = 150

    #classes = ['n02119789', 'n03773504', 'n04254680', 'n04429376', 'n04507155']
    #If you don't provide synsets, it will take images from all the sub-folders in images_dir
    print "Loading images from provided directories... "
    names, images, labels = loadImageNetFiles(images_dir, numberImagesPerClass)#, classes)

    #Preprocess images
    preprocessedImages = net.preprocess_images(images)

    #Forward pass to extract features and predict labels
    print "Performing a forward pass on all the images.... "
    probs, features = net.get_probs_and_features(preprocessedImages, extractionLayerName)

    predictedProbsTop5 =[np.argsort(x)[-1:-5:-1] for x in probs]
    
    predictedLabelsTop5 = []
    for predictions in predictedProbsTop5:
    	predictions = [idx_to_synset[x] for x in predictions]
    	predictedLabelsTop5.append(predictions)


    predictedLabelsTop1 = [predictions[0] for predictions in predictedLabelsTop5]

    net.get_precision(labels, predictedLabelsTop5)

    #Apply tsne
    
    perplexity = 10.0
    tolerance = 1e-5
    iterations = 800

    features = np.array(features)

    P = tsne.computeProbabilities(features, perplexity, tolerance)
    positions = tsne.computeMapPoints(P, iterations)

    mappedLabels = net.synsets_to_words([synset_to_class[x] for x in labels])
    mappedPredictedLabels = net.synsets_to_words([synset_to_class[x] for x in predictedLabelsTop1])

    tsne.showPoints(positions[-1], mappedLabels)
    tsne.showPoints(positions[-1], mappedPredictedLabels)

    tsne.showMovie(positions, mappedLabels)

    tsne.imagesPlot(images, positions[-1])
    

if __name__=='__main__':
    main(sys.argv[1:])
