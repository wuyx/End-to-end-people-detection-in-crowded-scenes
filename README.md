<img src=http://russellsstewart.com/s/ReInspect_output.jpg></img>

# ReInspect
ReInspect is an neural network extension to Overfeat-GoogLeNet in Caffe.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details.

## Installation
ReInspect depends on <a href="http://github.com/bvlc/caffe" target="_blank">Caffe</a> and requires
the <a href="http://apollocaffe.com">ApolloCaffe</a> pull request. With ApolloCaffe installed, you can run ReInspect with:

    $ git clone http://github.com/russell91/reinspect
    $ cd reinspect
    $ python train.py --config config.json --gpu -1

Data should be placed in /path/to/reinspect/data/ and can be found <a href="http://datasets.d2.mpi-inf.mpg.de/brainwash/brainwash.tar">here</a>.

## Demo
We provide a <a href="https://github.com/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb" target="_blank">notebook</a>
to visualize the performance of the model. The data includes a pretrained model, so you can run this notebook on your own machine without training.

<img src=http://russellsstewart.com/s/ReInspect.jpg></img>

## Running on your own data

The easiest way to run on your own data is to resize your images to 480x640 and provide labels for each object in each image with the idl text files.

Once you have verified that you can train on 480x640 images, you can also modify the image width and height options in config.json. We recommend you choose an image size which is an integer multiple of 32, and then modify the (15, 20) grid to (image_height / 32, image_width / 32).
