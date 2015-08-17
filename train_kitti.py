import numpy as np
from scipy.misc import imread
import random
import subprocess
import imp
import os
import random
import h5py
import pickle
from annolist_to_hdf5 import image_to_h5, annotation_to_h5, load_data_mean
from annolist_jitter import annotation_jitter
from annotation.annolist.python import AnnotationLib as al

import apollocaffe
from apollocaffe import layers
from apollocaffe.models import googlenet
from apollocaffe.layers import (Power, LstmUnit, Convolution, NumpyData,
    Transpose, Filler, SoftmaxWithLoss, Softmax, Concat, Dropout,
    InnerProduct)

def get_hyper():
    hyper = {}
    hyper["max_len"] = 5

    hyper["pal_mean"] = "./brainwash_27_13_24_train_lmdb.npy"
    hyper["train_pal"] = "/shared/u/hdd/brainwash/cafe_brainwash/brainwash_train.idl"
    hyper["test_pal"] = "/shared/u/hdd/brainwash/cafe_brainwash/brainwash_test.idl"

    hyper["deploy_batch_size"] = 300
    hyper["lstm_num_cells"] = 250
    hyper["googlenet_output_size"] = 1024
    hyper["dropout_ratio"] = 0.15 
    hyper["init_range"] = 0.1
    hyper["hungarian_loss_weight"] = 0.03
    hyper["hungarian_permute_matches"] = True
    hyper["hungarian_match_ratio"] = 0.5
    hyper["googlenet_weight_lr_mult"] = 1
    hyper["googlenet_weight_decay_mult"] = 1
    hyper["googlenet_bias_lr_mult"] = 2

    hyper["momentum"] = 0.5
    hyper["base_lr"] = 0.4 * (1. - hyper["momentum"])
    hyper["weight_decay"] = 0
    hyper["display_interval"] = 100
    hyper["max_iter"] = 2000000
    hyper["clip_gradients"] = 0.1
    hyper["snapshot_interval"] = 10000
    hyper["snapshot_prefix"] = "/deep/u/stewartr/snapshots/kitti%d_" % 0
    hyper["stepsize"] = 100000
    hyper["gamma"] = 0.8
    hyper["random_seed"] = 2
    hyper["test_interval"] = 100
    hyper["test_iter"] = 10
    hyper["graph_interval"] = 500
    hyper["graph_prefix"] = "/deep/u/stewartr/snapshots/%d" % 0
    hyper["weights"] = googlenet.weights_file()
    return hyper

def load_train_list_al(alfile):
    for al in alfile:
        yield al 

cell_height = 15
cell_width = 20
full_width = 640
full_height = 480
def load_train_list_pal(palfile, data_mean):
    annolist = al.parse(palfile)
    anno = [x for x in annolist]
    for x in anno:
        x.imageName = os.path.join(os.path.dirname(os.path.realpath(palfile)), x.imageName)
    while True:
        random.shuffle(anno)
        for a in anno:
            try:
                I, jit_a = annotation_jitter(a, target_width=full_width, target_height=full_height) 
            except:
                print 'problem: ', a
                continue
            #jit_a = a
            #I = imread(a.imageName)
            image = image_to_h5(I, data_mean, image_scaling=1.0)
            boxes, box_flags = annotation_to_h5(jit_a, cell_width, cell_height)
            yield {"image": image, "boxes": boxes, "box_flags": box_flags}

def load_train_list(train_list_txt):
    train_list = open(train_list_txt, "rb")
    lines = [x.strip() for x in train_list.readlines()]

    while True:
        # shuffle the list of h5 files
        random.shuffle(lines)
        for line in lines:
            yield line
    
def get_pickle(dir, pdict):
    if dir not in pdict:
        fpickle = subprocess.check_output("find %s -name *.pkl" % dir, shell=True).strip()
        h5_dict = dict(pickle.load(open(fpickle, 'rb')))
        pdict[dir] = h5_dict
    return pdict[dir]

def get_h5png(line_gen):
    h5file = line_gen.next()
    dir = '/'.join(h5file.split('/')[:-1])
    if not hasattr(get_h5png, "pdict"):
        get_h5png.pdict = dict()
    png = get_pickle(dir, get_h5png.pdict)[h5file]
    
    return (h5file, png)


def forward(net, input_data, hyper, deploy=False):
    net.clear_forward()
    if deploy:
        image = np.array(input_data)
    else:
        image = np.array(input_data["image"])
        box_flags = np.array(input_data["box_flags"])
        boxes = np.array(input_data["boxes"])

    net.f(NumpyData("image", data=image))

    google_layers = googlenet.googlenet_layers()
    google_layers[0].p.bottom[0] = "image"
    for layer in google_layers:
        if "loss" in layer.p.name:
            continue
        net.f(layer)
        if layer.p.name == "inception_5b/output":
            break

    net.f(Convolution("post_fc7_conv", bottoms=["inception_5b/output"],
        param_lr_mults=[1., 2.], param_decay_mults=[0., 0.], num_output=1024, kernel_dim=(1, 1),
        weight_filler=Filler("gaussian", 0.005),
        bias_filler=Filler("constant", 0.)))
    net.f(Power("lstm_fc7_conv", scale=0.01, bottoms=["post_fc7_conv"]))
    net.f(Transpose("lstm_input", bottoms=["lstm_fc7_conv"]))

    if not deploy:
        old_shape = list(box_flags.shape)
        new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        net.f(NumpyData("box_flags", data=np.reshape(
            box_flags, new_shape)))

        old_shape = list(boxes.shape)
        new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        net.f(NumpyData("boxes", data=np.reshape(
            boxes, new_shape)))

    net.f(NumpyData("lstm_hidden_seed", np.zeros((net.blobs["lstm_input"].shape[0], hyper["lstm_num_cells"]))))
    net.f(NumpyData("lstm_mem_seed", np.zeros((net.blobs["lstm_input"].shape[0], hyper["lstm_num_cells"]))))

    filler = Filler("uniform", hyper["init_range"])
    bias_filler = Filler("constant", 0)
    score_concat_bottoms = []
    bbox_concat_bottoms = []
    for step in range(hyper["max_len"]):
        if step == 0:
            hidden_bottom = "lstm_hidden_seed"
            mem_bottom = "lstm_mem_seed"
        else:
            hidden_bottom = "lstm_hidden%d" % (step - 1)
            mem_bottom = "lstm_mem%d" % (step - 1)

        net.f(Concat("concat%d" % step, bottoms=["lstm_input", hidden_bottom]))
        net.f(LstmUnit("lstm%d" % step, num_cells = hyper["lstm_num_cells"],
            weight_filler=filler,
            param_names=["input_value", "input_gate", "forget_gate", "output_gate"],
            bottoms=["concat%d" % step, mem_bottom],
            tops=["lstm_hidden%d" % step, "lstm_mem%d" % step]))
        net.f(Dropout("dropout%d" % step, hyper["dropout_ratio"],
            bottoms=["lstm_hidden%d" % step]))
        net.f(InnerProduct("ip_conf%d" % step, 2, bottoms=["dropout%d" % step],
            output_4d=True,
            weight_filler=filler))
        net.f(InnerProduct("ip_bbox_unscaled%d" % step, 4, bottoms=["dropout%d" % step],
            output_4d=True,
            weight_filler=filler))
        net.f(Power("ip_bbox%d" % step, scale=100, bottoms=["ip_bbox_unscaled%d" % step]))
        net.f(Softmax("ip_soft_conf%d" % step, bottoms = ["ip_conf%d"%step]))

        score_concat_bottoms.append("ip_conf%d" % step)
        bbox_concat_bottoms.append("ip_bbox%d" % step)

    net.f(Concat("score_concat", bottoms=score_concat_bottoms, concat_dim=2))
    net.f(Concat("bbox_concat", bottoms=bbox_concat_bottoms, concat_dim=2))

    if not deploy:
        net.f('''
            name: "hungarian"
            type: "HungarianLoss"
            bottom: "bbox_concat"
            bottom: "boxes"
            bottom: "box_flags"
            top: "hungarian"
            top: "box_confidences"
            top: "box_assignments"
            loss_weight: 0.03
            hungarian_loss_param {
              match_ratio: 0.5
              permute_matches: true
            }''')
        net.f(SoftmaxWithLoss("box_loss", bottoms=["score_concat", "box_confidences"]))

    bbox = [np.array(net.blobs["ip_bbox%d" % j].data) for j in range(hyper["max_len"])]
    conf = [np.array(net.blobs["ip_soft_conf%d" % j].data) for j in range(hyper["max_len"])]

    if deploy:
        bbox = [np.array(net.blobs["ip_bbox%d" % j].data) for j in range(hyper["max_len"])]
        conf = [np.array(net.blobs["ip_soft_conf%d" % j].data) for j in range(hyper["max_len"])]
        return (bbox, conf) 
    else:
        return None

def train(hyper):
    net = apollocaffe.ApolloNet()

    image_mean = load_data_mean(hyper["pal_mean"], full_width, full_height, image_scaling=1.0)
    input_gen = load_train_list_pal(hyper["train_pal"], image_mean)
    input_gen_test = load_train_list_pal(hyper["test_pal"], image_mean)

    forward(net, input_gen.next(), hyper)
    net.draw_to_file("/tmp/lstm_detect.png")

    net.load(hyper["weights"])

    train_loss_hist = []
    test_loss_hist = []
    loggers = [
        apollocaffe.loggers.TrainLogger(100),
        apollocaffe.loggers.TestLogger(100),
        apollocaffe.loggers.SnapshotLogger(10000, '/shared/u/hdd/snapshots/reinspect'),
        ]
    for i in range(hyper["start_iter"], hyper["max_iter"]):
        if i % hyper["test_interval"] == 0:
            net.phase = 'test'
            for j in range(hyper["test_iter"]):
                forward(net, input_gen_test.next(), hyper, False)
                test_loss_hist.append(net.loss)
            net.phase = 'train'
        forward(net, input_gen.next(), hyper)
        train_loss_hist.append(net.loss)
        net.backward()
        lr = (hyper["base_lr"] * (hyper["gamma"])**(i // hyper["stepsize"]))
        net.update(lr=lr, momentum=hyper["momentum"],
            clip_gradients=hyper.get("clip_gradients", -1))
        for logger in loggers:
            logger.log(i, {'train_loss': train_loss_hist, 'test_loss': test_loss_hist,
                'apollo_net': net, 'start_iter': 0})

def main():
    parser = apollocaffe.base_parser()
    args = parser.parse_args()
    hyper = get_hyper()
    assert hyper["weights"]
    if args.weights is not None:
        hyper["weights"] = args.weights
    hyper["start_iter"] = args.start_iter
    apollocaffe.set_random_seed(hyper["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    train(hyper)

if __name__ == "__main__":
    main()
