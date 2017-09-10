# CNN_Visualization
Implementation of visualization techniques for CNN in Caffe (t-sne, DeconvNet, Image occlusions)

#### Requirements:

- **Caffe** and **pyCaffe**, [Installation guide](http://caffe.berkeleyvision.org/installation.html).


- numpy, scikit-image, sklearn, skdata


## t-sne:
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



## OCCLUSION:
The file cnn_occlusion.py is an implementation of the occlusion technics described in the section **4.2** of the following paper: [Visualizing and Understanding Convolutional Networks](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf).

### How to use
In the following, the procedure to use **caffenet** will be described. It is possible to use different models by changing the default parameters. The **caffenet** prototxt is already included in the **caffe** installation. 

To download the caffemodel file in the proper place run:

        $ wget -O path/to/caffe/models/bvls_reference_caffenet/bvlc_reference_caffenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

Choose a class from the file **synset_words.txt** and download at least one image from [Image-Net](www.image-net.org).
Suppose the name of the image is *n01440764_18.JPEG*. It's important to don't change the name of the image since it contains the synset reference.

To test the code:

        $ python cnn_occlusion.py /path/to/n01440764_18.JPEG  
        
### Parameters:


Parameter | Description
------------ | -------------
image_path, PATH | The image path 
--weights, -w PATH | The model path (default: /path/to/caffenet_model)
--prototxt, -p PATH | The prototxt path (default: /path/to/caffenet_prototxt)
--layer, -l PATH | Extraction layer (default: pool5)
--gpu, -g INT | GPU number to use (default: -1, aka cpu_mode)
--batch_size INT| The batch size (default: 1)
--stride INT | The stride of the applied mask (default: 100)
--mask_size INT | The length of the side of the square mask (default: 100)

        


