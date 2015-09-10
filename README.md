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

## Evaluation
To evaluate ReInspect, we provide an <a href="https://github.com/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb" target="_blank">ipython notebook</a>
to visualize the performance of the model. 

<img src=http://russellsstewart.com/s/ReInspect.jpg></img>
