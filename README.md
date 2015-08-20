<img src=http://russellsstewart.com/s/ReInspect_output.jpg></img>

# ReInspect
ReInspect is an neural network extension to Overfeat-GoogLeNet in Caffe.
It is designed for high performance object detection in images with heavily overlapping instances.
See <a href="http://arxiv.org/abs/1506.04878" target="_blank">the paper</a> for details.

## Installation
ReInspect depends on <a href="http://github.com/bvlc/caffe" target="_blank">Caffe</a> and requires
the <a href="http://github.com/Russell91/apollocaffe">ApolloCaffe</a> pull request. 

    $ git clone http://github.com/russell91/apollocaffe
    $ cp <your caffe Makefile.config> apollocaffe/Makefile.config
    $ cd apollocaffe && make -j8
    $ export PYTHONPATH=/path/to/apollocaffe/python:$PYTHONPATH
    $ export LD_LIBRARY_PATH=/path/to/apollocaffe/build/lib:$LD_LIBRARY_PATH
    
With ApolloCaffe installed, you can run ReInspect with:

    $ git clone http://github.com/russell91/reinspect
    $ cd reinspect
    $ python train.py --config config.json --gpu -1

## Evaluation
To evaluate ReInspect we have provided an ipython notebook that can be used to visualize
the performance of the model.

The evaluation script depends on the python munkres library and ipython notebook.

To run the notebook cd into the reinspect directory and run 

    $ ipython notebook "--ip=*"

Open your favourite browser and navigate to "localhost:8888/notebooks/evaluation_reinspect.ipynb".
From there you should be able to evaluate the reinspect model by running each of the ipython notebook boxes.

<img src=http://russellsstewart.com/s/ReInspect.jpg></img>
