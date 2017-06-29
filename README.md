# CNN_Visualization
=========
Implementation of visualization techniques for CNN in Caffe (t-sne, DeconvNet, Image occlusions)

#### Requirements:

- **Caffe** and **pyCaffe**

- numpy, sklearn, matplotlib





t-sne:
-------------

The file tsne.py contains an implementation of t-Stochastic Neighbor Embedding as described in the following paper: [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf).

It's possible to test this implementation stand-alone on the well known mnist dataset. It's necessary to download data and labels from [here](https://github.com/azinik/java-deeplearning/tree/master/deeplearning4j-core/src/main/resources) and then simply run

        $ python tsne.py

To test how the implementation works with features extracted from a CNN, it's necessary first of all to download some images from ImageNet and extract the synset folders in a directory  called "image_dir".

In order to get the mapping from network output to synsets it's necessary to run

        $ ./path/to/caffe/data/ilsvrc12/data/get_ilsvrc_aux.sh

It's possible to test the implementation on the standard CaffeNet network. To download the caffemodel file in the proper place it's necessary to run

        $ wget -O path/to/caffe/models/bvls_reference_caffenet/bvlc_reference_caffenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

Finally it's possible to launch the main script.

        $ python tsneCNN.py -i image_dir -g

NOTE: The -g flag has to be used only if we want to run the script in gpu mode.
