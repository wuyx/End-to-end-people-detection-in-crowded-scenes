import numpy as np
import json
import subprocess
import os
import random
import pickle
import apollocaffe
from apollocaffe.models import googlenet
from apollocaffe.layers import (Power, LstmUnit, Convolution, NumpyData,
    Transpose, Filler, SoftmaxWithLoss, Softmax, Concat, Dropout,
    InnerProduct)

from annolist_to_hdf5 import image_to_h5, annotation_to_h5, load_data_mean
from annolist_jitter import annotation_jitter
from annotation.annolist.python import AnnotationLib as al

cell_height = 15
cell_width = 20
full_width = 640
full_height = 480
def load_train_list_pal(palfile, data_mean):
    annolist = al.parse(palfile)
    anno = [x for x in annolist]
    for x in anno:
        x.imageName = os.path.join(
            os.path.dirname(os.path.realpath(palfile)), x.imageName)
    while True:
        random.shuffle(anno)
        for a in anno:
            try:
                I, jit_a = annotation_jitter(
                    a, target_width=full_width, target_height=full_height)
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
        fpickle = subprocess.check_output(
            "find %s -name *.pkl" % dir, shell=True).strip()
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


def forward(net, input_data, net_config, deploy=False):
    net.clear_forward()
    if deploy:
        image = np.array(input_data["image"])
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
        param_lr_mults=[1., 2.], param_decay_mults=[0., 0.],
        num_output=1024, kernel_dim=(1, 1),
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

    net.f(NumpyData("lstm_hidden_seed",
        np.zeros((net.blobs["lstm_input"].shape[0], net_config["lstm_num_cells"]))))
    net.f(NumpyData("lstm_mem_seed",
        np.zeros((net.blobs["lstm_input"].shape[0], net_config["lstm_num_cells"]))))

    filler = Filler("uniform", net_config["init_range"])
    score_concat_bottoms = []
    bbox_concat_bottoms = []
    for step in range(net_config["max_len"]):
        if step == 0:
            hidden_bottom = "lstm_hidden_seed"
            mem_bottom = "lstm_mem_seed"
        else:
            hidden_bottom = "lstm_hidden%d" % (step - 1)
            mem_bottom = "lstm_mem%d" % (step - 1)

        net.f(Concat("concat%d" % step, bottoms=["lstm_input", hidden_bottom]))
        net.f(LstmUnit("lstm%d" % step, net_config["lstm_num_cells"],
            weight_filler=filler,
            param_names=["input_value", "input_gate",
                         "forget_gate", "output_gate"],
            bottoms=["concat%d" % step, mem_bottom],
            tops=["lstm_hidden%d" % step, "lstm_mem%d" % step]))
        net.f(Dropout("dropout%d" % step, net_config["dropout_ratio"],
            bottoms=["lstm_hidden%d" % step]))
        net.f(InnerProduct("ip_conf%d" % step, 2, bottoms=["dropout%d" % step],
            output_4d=True,
            weight_filler=filler))
        net.f(InnerProduct("ip_bbox_unscaled%d" % step, 4,
            bottoms=["dropout%d" % step], output_4d=True, weight_filler=filler))
        net.f(Power("ip_bbox%d" % step, scale=100,
            bottoms=["ip_bbox_unscaled%d" % step]))
        net.f(Softmax("ip_soft_conf%d" % step, bottoms=["ip_conf%d"%step]))

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
        net.f(SoftmaxWithLoss("box_loss",
            bottoms=["score_concat", "box_confidences"]))

    bbox = [np.array(net.blobs["ip_bbox%d" % j].data)
        for j in range(net_config["max_len"])]
    conf = [np.array(net.blobs["ip_soft_conf%d" % j].data)
        for j in range(net_config["max_len"])]

    if deploy:
        bbox = [np.array(net.blobs["ip_bbox%d" % j].data)
                for j in range(net_config["max_len"])]
        conf = [np.array(net.blobs["ip_soft_conf%d" % j].data)
                for j in range(net_config["max_len"])]
        return (bbox, conf)
    else:
        return None

def train(config):
    net = apollocaffe.ApolloNet()

    data_config = config["data"]
    image_mean = load_data_mean(data_config["pal_mean"],
        full_width, full_height, image_scaling=1.0)
    input_gen = load_train_list_pal(data_config["train_pal"], image_mean)
    input_gen_test = load_train_list_pal(data_config["test_pal"], image_mean)

    forward(net, input_gen.next(), config["net"])
    net.draw_to_file("/tmp/lstm_detect.png")

    solver = config["solver"]
    net.load(solver["weights"])

    train_loss_hist = []
    test_loss_hist = []
    logging = config["logging"]
    loggers = [
        apollocaffe.loggers.TrainLogger(logging["display_interval"]),
        apollocaffe.loggers.TestLogger(solver["test_interval"]),
        apollocaffe.loggers.SnapshotLogger(logging["snapshot_interval"],
            logging["snapshot_prefix"]),
        ]
    for i in range(solver["start_iter"], solver["max_iter"]):
        if i % solver["test_interval"] == 0:
            net.phase = 'test'
            for _ in range(solver["test_iter"]):
                forward(net, input_gen_test.next(), config["net"], False)
                test_loss_hist.append(net.loss)
            net.phase = 'train'
        forward(net, input_gen.next(), config["net"])
        train_loss_hist.append(net.loss)
        net.backward()
        lr = (solver["base_lr"] * (solver["gamma"])**(i // solver["stepsize"]))
        net.update(lr=lr, momentum=solver["momentum"],
            clip_gradients=solver["clip_gradients"])
        for logger in loggers:
            logger.log(i, {'train_loss': train_loss_hist,
                'test_loss': test_loss_hist,
                'apollo_net': net, 'start_iter': 0})

def main():
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    assert config["solver"]["weights"]
    if args.weights is not None:
        config["solver"]["weights"] = args.weights
    config["solver"]["start_iter"] = args.start_iter
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(args.gpu)
    apollocaffe.set_cpp_loglevel(args.loglevel)

    train(config)

if __name__ == "__main__":
    main()
