import os
os.environ['GLOG_minloglevel'] = '3'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/home/ale/libs/caffe/python")

import caffe
import argparse
import itertools
import caffe_utils

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


def backspace(n):
    sys.stdout.write('\r'+n)
    sys.stdout.flush()


def apply_mask_iterator(img, mask_size=20, stride=1, batch_size=100):
    half_mask_size = int(mask_size/2)
    
    x_max_img = img.shape[0]
    y_max_img = img.shape[1]
    x_min_img = 0
    y_min_img = 0

    batch = []
    positions = []

    x_range = range(x_min_img, x_max_img, stride)
    y_range = range(y_min_img, y_max_img, stride)

    total_samples = len(x_range)*len(y_range)

    for idx, (x, y) in enumerate(itertools.product(x_range, y_range)):

        x_min = max(x - half_mask_size, x_min_img)
        x_max = min(x + half_mask_size, x_max_img)

        y_min = max(y - half_mask_size, y_min_img)
        y_max = min(y + half_mask_size, y_max_img)

        new_img = img.copy() 
        new_img[x_min:x_max, y_min:y_max] = [0, 0, 0]
        batch.append(new_img)
        positions.append([x,y])

        if len(batch) % batch_size == 0 or idx == total_samples - 1:
            yield batch, positions, total_samples
            batch = []
            positions = []


def check_positive(value):
    try: 
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("%s is not an int value" % value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is not a positive int value" % value)
    return ivalue


def main():

    # retrieving files path
    pycaffe_path = os.path.dirname(caffe.__file__)
    caffe_path = os.path.normpath(os.path.join(pycaffe_path, '../../'))
    mean_path = os.path.join(pycaffe_path, 'imagenet/ilsvrc_2012_mean.npy')
    synsets_num_path = os.path.join(os.getcwd(), 'synsets.txt')
    synsets_to_class_path = os.path.join(os.getcwd(), 'synset_words.txt')

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
        	synset_class = line.strip().split()
        	synset = synset_class[0]
        	class_ = " ".join(synset_class[1:])
        	synset_to_class[synset] = class_
        	class_to_synset[class_] = synset

    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', metavar='PATH', type=str,
                        help="Input image path, an ImageNet one is required.")
    parser.add_argument("-w", "--weights", metavar='PATH', type=str, 
    					help="the model file, (default: %(default)s).", 
                        default=os.path.join(caffe_path, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'))
    parser.add_argument("-p", "--prototxt", metavar='PATH', type=str, 
    					help="prototxt file, (default: %(default)s).",
                        default=os.path.join(caffe_path, 'models/bvlc_reference_caffenet/deploy.prototxt'))
    parser.add_argument("-l", "--layer", default='pool5', metavar='layer_name', type=str,
                        help="Extraction layer, (default: %(default)s)")
    parser.add_argument("-g", "--gpu", default=-1, metavar='INT', type=int,
                        help="GPU number, (default: %(default)s aka disabled)")
    parser.add_argument("--batch_size", type=int, default=1, metavar='INT',
                        help="Batch size, (default: %(default)s).")
    parser.add_argument("--stride", type=check_positive, default=20, metavar='INT',
                        help="The stride of the applied mask, (default: %(default)s).")
    parser.add_argument("--mask_size", type=check_positive, default=50, metavar='INT',
                        help="The length of the side of the square mask, (default: %(default)s).")
    args = parser.parse_args()

    model_filename = args.prototxt
    weight_filename = args.weights
    image_path = args.image_path
    batch_size = args.batch_size
    extraction_layer = args.layer
    stride = args.stride
    mask_size = args.mask_size
    
    # setting the mode, the default is cpu mode
    caffe.set_mode_cpu()

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
    synset = os.path.basename(image_path).split('_')[0]
    
    # preprocessing and extracting most active filter  
    preprocessed_img = net.preprocess_images([img])
    img_features = net.get_features(preprocessed_img, extraction_layer)

    # getting the most active filter index for the image
    most_active_filter = net.get_most_active_filters(img_features)[0][0]

    images_features = []
    synset_probabilities = []
    predicted_idxs = []
    all_positions = []
    true_synset_idx = synset_to_idx[synset]

    print '####################################' 
    print 'True synset: ', synset
    print 'True class: ', synset_to_class[synset]
    print '####################################'

    # the mask is applied before the image preprocessing
    iterator_ = apply_mask_iterator(img, mask_size=mask_size, stride=stride, batch_size=batch_size)

    for masked_images, positions, total_samples in iterator_:

        # storing the central position of the applied mask
        all_positions.extend(positions)

        # preprocessing the image according to the network input
        preprocessed_images = net.preprocess_images(masked_images)

        # extracting the softmax output and the most active convolutional features from the extraction layer
        probs, features = net.get_probs_and_features(preprocessed_images, extraction_layer, most_active_filter)
        images_features.extend(features)
        
        synset_probabilities.extend([x[true_synset_idx] for x in probs])
        best_synsets_idxs = [np.argsort(x)[-1] for x in probs]
        predicted_idxs.extend(best_synsets_idxs)
        
        to_print = '{} of {}'.format(len(images_features), total_samples)
        
        backspace(to_print)

    print 

    # heat_map of probability of the true class, 
    heat_map_size = img.shape[:2]

    # initializing heatmaps
    heat_map_probs = np.zeros(heat_map_size)
    heat_map_features = np.zeros(heat_map_size)
    heat_map_synsets = np.zeros(heat_map_size)
    heat_map_num = np.zeros(heat_map_size)

    # filling heatmaps
    for [x, y], prob, feature, predicted_idx in zip(all_positions, synset_probabilities, images_features, predicted_idxs):
        heat_map_probs[x, y] = prob
        heat_map_features[x, y] = np.mean(feature)
        heat_map_synsets[x, y] = predicted_idx
        heat_map_num[x, y] = 1

    # deleting empty rows and columns
    heat_map_probs = np.nan_to_num(np.divide(heat_map_probs, heat_map_num))
    means_0 = np.mean(heat_map_num, axis=1)
    heat_map_num = np.delete(heat_map_num, np.where(means_0 == 0)[0], axis=0)
    means_1 = np.mean(heat_map_num, axis=0)
    heat_map_num = np.delete(heat_map_num, np.where(means_1 == 0)[0], axis=1)
    
    heat_map_probs = np.delete(heat_map_probs, np.where(means_0 == 0)[0], axis=0)
    heat_map_probs = np.delete(heat_map_probs, np.where(means_1 == 0)[0], axis=1)

    heat_map_features = np.delete(heat_map_features, np.where(means_0 == 0)[0], axis=0)
    heat_map_features = np.delete(heat_map_features, np.where(means_1 == 0)[0], axis=1)
    max_feature = np.max(heat_map_features)
    min_feature = np.min(heat_map_features)

    heat_map_features = (heat_map_features - min_feature)/(max_feature - min_feature)
    heat_map_synsets = np.delete(heat_map_synsets, np.where(means_0 == 0)[0], axis=0)
    heat_map_synsets = np.delete(heat_map_synsets, np.where(means_1 == 0)[0], axis=1)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.subplots_adjust()

    ax1.imshow(img)
    ax1.set_title('Original image')

    cmap = plt.get_cmap('YlOrRd')
    img = ax2.imshow(heat_map_probs, cmap=cmap, vmin=0, vmax=1, interpolation='none')
    ax2.axis('off')
    ax2.set_title('Classifier, probability of correct class')
    plt.colorbar(img, ax=ax2, fraction=0.046, pad=0.04)

    img = ax3.imshow(heat_map_features, cmap=cmap, interpolation='none')
    ax3.axis('off')
    ax3.set_title('Strongest feature map')
    plt.colorbar(img, ax=ax3, fraction=0.046, pad=0.04)

    norm = Normalize(vmin=0, vmax=len(idx_to_synset))
    
    cmap = plt.get_cmap('rainbow')
    ax4.imshow(heat_map_synsets, cmap=cmap, interpolation='none', norm=norm)
    ax4.axis('off')
    ax4.set_title('Classifier, most probable class')
    synsets_set = list(set(heat_map_synsets.flatten().tolist()))
    class_set = [synset_to_class[idx_to_synset[x]].split(',')[0] for x in synsets_set]
    #synsets_set = [x for x in xrange(0,999,1000/len(synsets_set))]
    colors = [cmap(norm(x)) for x in synsets_set]
    handles = []
    for synset_id, color in zip(synsets_set, colors):
        handles.append(Rectangle((0,0),1,1, color=list(color[:3])))
    box = ax4.get_position()
    ax4.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax4.legend(handles, class_set, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)

    plt.show()


if __name__=='__main__':
    main()
