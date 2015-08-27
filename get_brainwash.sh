#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

mkdir -p data && cd data
wget http://russellsstewart.com/s/reinspect/brainwash_mean.npy
wget http://russellsstewart.com/s/reinspect/brainwash_800000.h5
wget http://datasets.d2.mpi-inf.mpg.de/brainwash/brainwash.tar
tar xf brainwash.tar

echo "Done."
