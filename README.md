# CNN_Visualization
Implementation of visualization techniques for CNN in Caffe (t-SNE, DeconvNet, Image occlusions)

#### Requirements:

- **Caffe** and **pyCaffe**, [Installation guide](http://caffe.berkeleyvision.org/installation.html).


- numpy, scikit-image, sklearn, skdata


## t-sne:
The file tsne.py contains an implementation of t-Stochastic Neighbor Embedding as described in the following paper: [Visualizing Data using t-SNE](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf).

It's possible to test this implementation stand-alone on the well known mnist dataset. It's necessary to download data and labels from [here](https://github.com/azinik/java-deeplearning/tree/master/deeplearning4j-core/src/main/resources), put them in the same folder of the source code and run


        $ python tsne.py

To test how the implementation works with features extracted from a CNN, we will use the file tsneCNN.py.


Parameter | Description
------------ | -------------
--input_im, -i PATH | Path to the folder containing the synset image folders
--weights, -w PATH | The model path (default: /path/to/caffenet_model)
--prototxt, -p PATH | The prototxt path (default: /path/to/caffenet_prototxt)
--gpu, -g | If this flag is used, the code will run in gpu mode
--net_type, -n STRING | The type of the CNN we have provided (options are resnet, googlenet, vggnet)

The simplest way to test the code consists in downloading the standard CaffeNet network using 

        $ wget -O path/to/caffe/models/bvls_reference_caffenet/bvlc_reference_caffenet.caffemodel http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel


Get the mapping from network output to synsets using

        $ ./path/to/caffe/data/ilsvrc12/data/get_ilsvrc_aux.sh

Download some synsets of images from ImageNet and place them in a folder called "image_dir".


Launch the script exploiting default parameters.

        $ python tsneCNN.py -i image_dir -g


It's possible to test different networks simply by downloading their prototxt and weights and providing the correct paths to the script.

The expected output consists in: 

- The 2-dimensional embedding of the extracted features (each image is represented as a "dot" coloured accoring to its real or predicted class). This allows to see how this unsupervised method performs really well regardless of the CNN prediction.

- A video showing how the embedding changes through the t-SNE iterations.

- The real images organized according to the embedding. This allows to see how t-SNE preserves local differences within the same class of images.

![alt text](https://github.com/albioTQ/CNN_Visualization/blob/master/t-sne.gif)




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


You can expect something like this:


![alt text](https://github.com/albioTQ/CNN_Visualization/blob/master/occlusion_output.png)

        
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

        


