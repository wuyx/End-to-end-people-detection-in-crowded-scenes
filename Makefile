SHELL := /bin/bash

.PHONY: all
all:
	@echo -e "Usage:\n\
	$$ make train         # train the network on the CPU \n\
	$$ make eval          # run the ipython notebook"

.PHONY: train
train: utils/stitch_wrapper.so data
	@echo Training...
	python train.py --config config.json --gpu 0

.PHONY: eval
eval: utils/stitch_wrapper.so
	ipython notebook evaluation_reinspect.ipynb

.PHONY: data
data: data/brainwash_mean.npy data/brainwash_800000.h5 data/brainwash.tar

data/brainwash_mean.npy:
	@mkdir -p data
	cd data && wget http://russellsstewart.com/s/reinspect/brainwash_mean.npy

data/brainwash_800000.h5:
	@mkdir -p data
	cd data && wget http://russellsstewart.com/s/reinspect/brainwash_800000.h5

data/brainwash.tar:
	@mkdir -p data
	cd data && wget http://datasets.d2.mpi-inf.mpg.de/brainwash/brainwash.tar
	cd data && tar xf brainwash.tar

.PHONY: clean
clean:
	rm -f utils/stitch_wrapper.so

utils/stitch_wrapper.so:
	cd utils && makecython++ stitch_wrapper.pyx "" "stitch_rects.cpp ./hungarian/hungarian.cpp"

