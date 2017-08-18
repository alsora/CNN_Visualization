import os
os.environ['GLOG_minloglevel'] = '3'

import sys
sys.path.append("/home/ale/libs/caffe/python")

import caffe
import argparse
import itertools
import caffe_utils

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib import cm

def backspace(n):
    sys.stdout.write('\r'+n)
    sys.stdout.flush()

def apply_mask(img, mask_size=70, stride=20):
    half_mask_size = int(mask_size/2)
    
    x_max = img.shape[0] - half_mask_size
    y_max = img.shape[1] - half_mask_size
    x_min = half_mask_size
    y_min = half_mask_size

    batch = []
    positions = []

    x_range = range(x_min, x_max + 1, stride)
    y_range = range(y_min, y_max + 1, stride)

    for x, y in itertools.product(x_range, y_range):

        new_img = img.copy() 
        new_img[x-half_mask_size:x+half_mask_size, y-half_mask_size:y+half_mask_size] = [0, 0, 0]
        batch.append(new_img)
        positions.append([x,y])

    return batch, positions

def apply_mask_iterator(img, mask_size=20, stride=1, batch_size=100):
    half_mask_size = int(mask_size/2)
    
    x_max_img = img.shape[0]
    y_max_img = img.shape[1]
    x_min_img = 0
    y_min_img = 0

    batch = []
    positions = []

    x_range = range(x_min_img, x_max_img, stride)
    y_range = range(x_min_img, x_max_img, stride)

    for x, y in itertools.product(x_range, y_range):

        x_min = max(x - half_mask_size, x_min_img)
        x_max = min(x + half_mask_size, x_max_img)

        y_min = max(y - half_mask_size, y_min_img)
        y_max = min(y + half_mask_size, y_max_img)

        new_img = img.copy() 
        new_img[x_min:x_max, y_min:y_max] = [0, 0, 0]
        batch.append(new_img)
        positions.append([x,y])

        if len(batch) % batch_size == 0:
            yield batch, positions
            batch = []
            positions = []


def main():

    pycaffe_path = os.path.dirname(caffe.__file__)
    caffe_path = os.path.normpath(os.path.join(pycaffe_path, '../../'))
    mean_path = os.path.join(pycaffe_path, 'imagenet/ilsvrc_2012_mean.npy')
    synsetsNum_path = os.path.join(caffe_path, 'data/ilsvrc12/synsets.txt')
    synsets_to_class_path = os.path.join(os.getcwd(), 'synset_words.txt')

    idx_to_synset = {}
    synset_to_idx = {}

    with open(synsetsNum_path, 'r') as fp:
        for idx, synset in enumerate(fp):
            synset = synset.strip()
            idx_to_synset[idx] = synset
            synset_to_idx[synset] = idx

    synset_to_class = {}
    class_to_synset = {}

    with open(synsets_to_class_path, 'r') as fp:
        for line in fp:
            [synset, class_] = line.strip().split('\t')
            synset_to_class[synset] = class_
            class_to_synset[class_] = synset

    caffe.set_mode_cpu()

    parser = argparse.ArgumentParser()

    parser.add_argument("-w", "--weights", help="caffemodel file, (default: caffenet)",
                        default=os.path.join(caffe_path, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'))
    parser.add_argument("-p", "--prototxt", help="prototxt file, (default: caffenet)",
                        default=os.path.join(caffe_path, 'models/bvlc_reference_caffenet/deploy.prototxt'))
    parser.add_argument("-i", "--image_path", required=True,
                        help="Input image path, an ImageNet one is required.")
    parser.add_argument("-l", "--layer", default='pool5',
                        help="Extraction layer, (default: pool5)")
    parser.add_argument("-g", "--gpu", default=-1,
                        help="GPU number, (default: -1, aka disabled)")
    parser.add_argument("--total_images", type=int, default=100,
                        help="Number of images to process")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")

    args = parser.parse_args()

    model_filename = args.prototxt
    weight_filename = args.weights
    image_path = args.image_path
    batch_size = args.batch_size
    extraction_layer = args.layer
    
    if args.gpu > -1:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)

    if os.path.isfile(model_filename):
        print 'Caffe model found.'
    else:
        print 'Caffe model NOT found...'
        sys.exit()

    # Loading net and utilities    
    net = caffe_utils.CaffeNet(model_filename, weight_filename, mean_path, batch_size=batch_size)
    
    # Loading image to process
    img = caffe.io.load_image(image_path)
    label = os.path.basename(image_path).split('_')[0]
    
    # preprocessing and extracting most active filter  
    preprocessed_img = net.preprocess_images([img])
    img_features = net.extract_features(preprocessed_img, extraction_layer)
    most_active_filter = net.get_most_active_filters(img_features)[0][0]

    images_features = []
    probs = []
    label_probabilities = []
    predicted_idxs = []
    all_positions = []

    # the mask is applied before the image preprocessing
    for masked_images, positions in apply_mask_iterator(img, mask_size=100, stride=1, batch_size=batch_size):

        num_of_images = len(masked_images)
        all_positions.extend(positions)

        preprocessed_images = net.preprocess_images(masked_images)
        images_features.extend(net.extract_features(preprocessed_images, extraction_layer, most_active_filter))
        
        probs.extend(net.get_probs(preprocessed_images))
        
        true_label_idx = synset_to_idx[label]
        label_probabilities.extend([x[true_label_idx] for x in probs[-num_of_images:]])

        best_labels_idxs = [np.argsort(x)[-1] for x in probs[-num_of_images:]]

        predicted_idxs.extend(best_labels_idxs)

        to_print = '{0} of {1}'.format(len(images_features), img.shape[0]*img.shape[1])
        backspace(to_print)


    # heat_map of probability of the true class
    heat_map_size = img.shape[0]

    heat_map_probs = np.zeros((heat_map_size, heat_map_size))
    heat_map_features = np.zeros((heat_map_size, heat_map_size))
    heat_map_labels = np.zeros((heat_map_size, heat_map_size))
    heat_map_num= np.zeros((heat_map_size, heat_map_size))

    cmap_labels = cm.autumn
    norm = Normalize(vmin=0, vmax=len(idx_to_synset))

    for [x, y], prob, feature, predicted_idx in zip(positions, label_probabilities, images_features, predicted_idxs):
        
        heat_map_probs[x, y] += prob
        heat_map_features[x, y] += np.mean(feature)
        heat_map_labels[x, y] += predicted_idx
        heat_map_num[x, y] += 1

    # heat_map_probs = np.nan_to_num(np.divide(heat_map_probs, heat_map_num))

    plt.imshow(heat_map_probs, cmap=cm.Blues, interpolation='nearest')
    plt.colorbar()

    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(heat_map_features, cmap=cm.Blues)

    plt.figure()
    plt.imshow(heat_map_labels, cmap=cmap_labels, norm=norm)
    plt.show()


if __name__=='__main__':
    main()
