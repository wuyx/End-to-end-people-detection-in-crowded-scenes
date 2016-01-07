<img src=http://russellsstewart.com/s/ReInspect_output.jpg></img>

# ReInspect
ReInspect is an neural network extension to Overfeat-GoogLeNet in Caffe.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details or the <a href="https://www.youtube.com/watch?v=QeWl0h3kQ24" target="_blank">video</a> for a demonstration.

## Installation & Demo

    $ # install the apollocaffe pull request on caffe. see apollocaffe.com
    $ git clone http://github.com/russell91/reinspect
    $ cd reinspect
    $ make eval

Running `make eval` will download the data and start a <a href="https://github.com/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb" target="_blank">notebook</a>
to visualize the performance of the model. If you want to train the model from scratch, you can use

    $ make train

<img src=http://russellsstewart.com/s/ReInspect.jpg></img>

## Running on your own data

The easiest way to run on your own data is to resize your images to 480x640 and provide labels for each object in each image with the idl text files.

Once you have verified that you can train on 480x640 images, you can also modify the image width and height options in config.json. We recommend you choose an image size which is an integer multiple of 32, and then modify the (15, 20) grid to (image_height / 32, image_width / 32).
