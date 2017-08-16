import os

os.environ['GLOG_minloglevel'] = '3'


import caffe
import sys
import argparse
import itertools
import numpy as np
import caffe_utils
import caffeCNN_utils as CNN

from random import randint
from dataset_utils import loadImageNetFiles
import matplotlib.pyplot as plt
from skimage.transform import resize

netLayers = {
    'caffenet': 'fc7',

   #'googlenet': 'inception_5b/output'
   'googlenet': 'loss3/classifier'
}

def apply_mask(img, mask_size=70, stride=20):
    half_mask_size = int(mask_size/2)
    x_max, y_max = [img.shape[0] - half_mask_size, img.shape[1] - half_mask_size]
    x_min, y_min = [half_mask_size, half_mask_size]

    batch = []
    positions = []
    idx = 0

    x_range = range(x_min, x_max + stride, stride)
    y_range = range(x_min, x_max + stride, stride)


    for x, y in itertools.product(x_range, y_range):

        new_img = img.copy() 
        new_img[x-half_mask_size:x+half_mask_size-1, y-half_mask_size:y+half_mask_size-1] = [0, 0, 0]
        batch.append(new_img)
        positions.append([x,y])

    return batch, positions


def main():

    pycaffe_path = os.path.dirname(caffe.__file__)
    caffe_path = os.path.normpath(os.path.join(pycaffe_path, '../../'))
    mean_path = os.path.join(pycaffe_path, 'imagenet/ilsvrc_2012_mean.npy')
    synsetsNum_path = os.path.join(caffe_path, 'data/ilsvrc12/synsets.txt')
    synsetsWords_path = os.path.join(os.getcwd(), 'synset_words.txt')

    idx_to_synset = {}
    synset_to_idx = {}

    with open(synsetsNum_path, 'r') as fp:
        for idx, synset in enumerate(fp):
            synset = synset.strip()
            idx_to_synset[idx] = synset
            synset_to_idx[synset] = idx

    caffe.set_mode_cpu()

    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="caffemodel file",
                        default=os.path.join(caffe_path, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'))
    parser.add_argument("-p", "--prototxt", help="prototxt file",
                        default=os.path.join(caffe_path, 'models/bvlc_reference_caffenet/deploy.prototxt'))
    parser.add_argument("-i", "--images_dir", default='images_dir',
                        help="Input images dir")
    parser.add_argument("-n", "--net_type", default='caffenet',
                        help="CNN type (resnet/googlenet/caffe_net")
    parser.add_argument("-g", "--gpu", default=-1,
                        help="GPU number")
    parser.add_argument("--total_images", type=int, default=100,
                        help="Number of images to process")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")

    args = parser.parse_args()

    model_filename = args.prototxt
    weight_filename = args.weights
    images_dir = args.images_dir
    cnn_type = args.net_type
    batch_size = args.batch_size
    total_images = args.total_images

    if args.gpu > -1:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)

    if os.path.isfile(model_filename):
        print 'Caffe model found.'
    else:
        print 'Caffe model NOT found...'
        sys.exit()


    net = caffe_utils.CaffeNet(model_filename, weight_filename, mean_path, batch_size=batch_size)
    extractionLayerName = netLayers[cnn_type]

    #If you don't provide synsets, it will take images from all the sub-folders in images_dir
    names, images, labels = loadImageNetFiles(images_dir, total_images)

    for idx, img in enumerate(images):
        img = resize(img, [227,227], order=1)
        masked_images, positions = apply_mask(img, stride=3, mask_size=100)
        #plt.imshow(masked_images[0])
        #plt.show()
        preprocessed_images = net.preprocess_images(masked_images)
        probs = net.get_probs(preprocessed_images)
        
        label_idx = synset_to_idx[labels[idx]]
        label_probabilities = [x[label_idx] for x in probs]

        heat_map_size = 30
        heat_map = np.zeros((heat_map_size, heat_map_size))
        heat_map_num = np.zeros((heat_map_size, heat_map_size))

        for [x, y], prob in zip(positions, label_probabilities):
            
            x = int(x/float(img.shape[0])*heat_map_size - 1)
            y = int(y/float(img.shape[1])*heat_map_size - 1)
            heat_map[x, y] += prob
            heat_map_num[x, y] += 1

        print heat_map
        print heat_map_num
        heat_map = np.nan_to_num(np.divide(heat_map, heat_map_num))
        print heat_map
        plt.imshow(heat_map, cmap=plt.cm.Blues, interpolation='nearest')
        plt.colorbar()

        plt.figure()
        plt.imshow(img)
        plt.show()


if __name__=='__main__':
    main()
