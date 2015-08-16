import numpy as np
import matplotlib; matplotlib.use('Agg', warn=False); import matplotlib.pyplot as plt
import logging
import argparse
import random
import subprocess
import json
import imp
import os
import h5py
import pickle
import apollocaffe
from annolist_to_hdf5 import image_to_h5, annotation_to_h5, load_data_mean
from annolist_jitter import annotation_jitter
import AnnotationLib as al
from utils import IouDetection
from apollocaffe import caffe_pb2
from apollocaffe import layers
from apollocaffe.models import googlenet
from apollocaffe.layers import Power, LstmUnit, Convolution, NumpyData, Transpose, Filler, SoftmaxWithLoss, Softmax, Concat

def load_train_list_al(alfile) :
    for al in alfile :
        yield al 

cell_height = 15
cell_width = 20
full_width = 640
full_height = 480
def load_train_list_pal(palfile, data_mean) :
    annolist = al.parse(palfile)
    anno = [r for r in annolist]
    while True:
        random.shuffle(anno)
        for a in anno:
            I, jit_a = annotation_jitter(a, target_width=full_width, target_height=full_height) 
            image = image_to_h5(I, data_mean, image_scaling=1.0)
            boxes, box_flags = annotation_to_h5(jit_a, cell_width, cell_height)
            yield {'image': image, 'boxes': boxes, 'box_flags': box_flags}

def load_train_list(train_list_txt) :
    train_list = open(train_list_txt, 'rb')
    lines = [x.strip() for x in train_list.readlines()]

    while True :
        # shuffle the list of h5 files
        random.shuffle(lines)
        for line in lines :
            yield line
    
def get_pickle(dir, pdict):
    if dir not in pdict:
        fpickle = subprocess.check_output('find %s -name *.pkl' % dir, shell=True).strip()
        h5_dict = dict(pickle.load(open(fpickle, 'rb')))
        pdict[dir] = h5_dict
    return pdict[dir]

def get_h5png(line_gen) :
    h5file = line_gen.next()
    dir = '/'.join(h5file.split('/')[:-1])
    if not hasattr(get_h5png, 'pdict'):
        get_h5png.pdict = dict()
    png = get_pickle(dir, get_h5png.pdict)[h5file]
    
    return (h5file, png)


def forward(net, h5file, hyper, deploy=False, input_str=False):
    if deploy :
        if input_str:
            with h5py.File(h5file, 'r') as f:
                image = np.array(f['image'])
        else:
            image = np.array(h5file)
    else :
        f = h5file
        image = np.array(f['image'])
        box_flags = np.array(f['box_flags'])
        boxes = np.array(f['boxes'])

    net.f(layers.NumpyData(name="image", data=image))

    google_layers = googlenet.googlenet_layers()
    google_layers[0].p.bottom[0] = "image"
    for layer in google_layers:
        if 'loss' in layer.p.name:
            continue
        net.f(layer)
        if layer.p.name == 'inception_5b/output':
            break

    net.f(layers.Convolution(name='post-fc7-conv', bottoms=['inception_5b/output'],
        param_lr_mults=[1., 2.], param_decay_mults=[0., 0.], num_output=1024, kernel_size=9, stride=2, pad=4,
        #param_lr_mults=[1., 2.], param_decay_mults=[0., 0.], num_output=1024, kernel_size=1,
        weight_filler=layers.Filler(type='gaussian', std=0.005),
        bias_filler=layers.Filler(type='constant')))
        
    net.f(layers.Power(name='lstm-fc7-conv', scale=0.01, bottoms=['post-fc7-conv']))
    net.f(layers.Transpose(name='lstm_input', bottoms=['lstm-fc7-conv']))

    if not deploy:
        old_shape = list(box_flags.shape)
        new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        net.f(layers.NumpyData(name='box_flags_mod', data=np.reshape(
            box_flags, new_shape)))

        old_shape = list(boxes.shape)
        new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        net.f(layers.NumpyData(name='boxes_mod', data=np.reshape(
            boxes, new_shape)))

    net.f(layers.DummyData(name='dummy_hidden', shape=(net.tops['lstm_input'].shape[0], hyper['lstm_num_cells'], 1, 1)))
    net.f(layers.DummyData(name='dummy_mem', shape=(net.tops['lstm_input'].shape[0], hyper['lstm_num_cells'], 1, 1)))

    filler = layers.Filler(type='uniform', min=-hyper['init_range'],
        max=hyper['init_range'])
    bias_filler = layers.Filler(type='constant')
    score_concat_bottoms = []
    bbox_concat_bottoms = []
    for step in range(hyper['max_len']):
        if step == 0:
            hidden_bottom = 'dummy_hidden'
            mem_bottom = 'dummy_mem'
        else:
            hidden_bottom = 'lstm_hidden%d' % (step - 1)
            mem_bottom = 'lstm_mem%d' % (step - 1)
        net.f(layers.Concat(name='concat%d' % step, bottoms=['lstm_input', hidden_bottom]))
        net.f(layers.Lstm(name=('lstm%d' % step), num_cells = hyper['lstm_num_cells'],
            weight_filler=filler, output_4d=True,
            param_names=['input_value', 'input_gate', 'forget_gate', 'output_gate'],
            bottoms=['concat%d' % step, mem_bottom],
            tops=['lstm_hidden%d' % step, 'lstm_mem%d' % step]))

        net.f(layers.Dropout(name=('dropout%d' % step), bottoms=['lstm_hidden%d' % step],
            dropout_ratio=hyper['dropout_ratio']))

        net.f(layers.InnerProduct(name=('ip_conf%d' % step), bottoms=['dropout%d' % step],
            output_4d=True, num_output=2,
            weight_filler=filler, bias_filler=bias_filler))#,
            #param_names=['ip_conf_weight', 'ip_conf_bias']))
        net.f(layers.InnerProduct(name=('ip_bbox_unscaled%d' % step), bottoms=['dropout%d' % step],
            output_4d=True, num_output=4,
            weight_filler=filler, bias_filler=bias_filler))#,
            #param_names=['ip_bbox_weight', 'ip_bbox_bias']))
        net.f(layers.Power(name=('ip_bbox%d' % step), bottoms=['ip_bbox_unscaled%d' % step], scale=400))

        net.f(layers.Softmax(name="ip_soft_conf%d" % step, bottoms = ['ip_conf%d'%step]))
        score_concat_bottoms.append('ip_conf%d' % step)
        bbox_concat_bottoms.append('ip_bbox%d' % step)
    net.f(layers.Concat(name='score_concat', bottoms=score_concat_bottoms, concat_dim=2))
    net.f(layers.Concat(name='bbox_concat', bottoms=bbox_concat_bottoms, concat_dim=2))

    if not deploy:
        loss = 0.
        hungarian_layer = caffe_pb2.LayerParameter()
        hungarian_layer.name = "hungarian"
        hungarian_layer.top.append(hungarian_layer.name)
        hungarian_layer.top.append("box_confidences")
        hungarian_layer.top.append("box_assignments")
        hungarian_layer.loss_weight.append(hyper['hungarian_loss_weight'])
        hungarian_layer.type = "HungarianLoss"
        hungarian_layer.hungarian_loss_param.permute_matches = hyper['hungarian_permute_matches']
        hungarian_layer.hungarian_loss_param.top_assignments = True
        hungarian_layer.hungarian_loss_param.match_ratio = hyper['hungarian_match_ratio']
        hungarian_layer.bottom.append('bbox_concat')
        hungarian_layer.bottom.append('boxes_mod')
        hungarian_layer.bottom.append('box_flags_mod')

        loss += net.f(layers.Unknown(hungarian_layer))
        loss += net.f(layers.SoftmaxWithLoss(name='box_loss', bottoms=['score_concat', 'box_confidences'],
            ignore_label=hyper['zero_symbol']))

    bbox = [np.array(net.tops['ip_bbox%d' % j].data) for j in range(hyper['max_len'])]
    conf = [np.array(net.tops['ip_soft_conf%d' % j].data) for j in range(hyper['max_len'])]

    if deploy :
        bbox = [np.array(net.tops['ip_bbox%d' % j].data) for j in range(hyper['max_len'])]
        conf = [np.array(net.tops['ip_soft_conf%d' % j].data) for j in range(hyper['max_len'])]
        return (None, bbox, conf) 
        #bbox_label = [np.array(boxes[0,:,:,i:i+1,:]) for i in range(hyper['max_len'])]
        #return (None, bbox_label, conf) 
    else :
        return (loss, None, None) 

def train(hyper):
    net = apollocaffe.Net()
    test_net = apollocaffe.Net()

    if hyper['use_pal']:
        img_height = full_height
        img_width = full_width
        image_mean = load_data_mean(hyper['pal_mean'], img_width, img_height, image_scaling = 1.0)
        h5file = load_train_list_pal(hyper['train_pal'], image_mean)
        h5filetest = load_train_list_pal(hyper['val_pal'], image_mean)
    else:
        h5file = load_train_list(hyper['train_h5list'])
        h5filetest = load_train_list(hyper['val_h5list'])
    forward(net, h5file.next(), hyper)
    net.draw_to_file('/tmp/lstm_detect.png')
    net.reset_forward()

    forward(test_net, h5file.next(), hyper)
    test_net.reset_forward()

    #print net.params.keys()
    #net.load(googlenet.weights_file())
    net.load(hyper['weights'])
    #net.load("/deep/u/ysavani/snapshots/brainwash/brainwash_250000.h5")
    train_loss_hist = []
    test_loss_hist = []

    for i in range(hyper['start_iter'], hyper['max_iter']):
        train_loss_hist.append(forward(net, h5file.next(), hyper)[0])
        net.backward()
        lr = (hyper['base_lr'] * (hyper['gamma'])**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper.get('clip_gradients', -1), weight_decay=hyper['weight_decay'])
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
        if i % hyper['snapshot_interval'] == 0 and i > 0:
            filename = '%s_%d.h5' % (hyper['snapshot_prefix'], i)
            logging.info('Saving net to: %s' % filename)
            net.save(filename)
        if i % hyper['graph_interval'] == 0 and i > 0:
            sub = 1000
            plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub])
            filename = '%strain_loss.jpg' % hyper['graph_prefix']
            logging.info('Saving figure to: %s' % filename)
            plt.savefig(filename)

            sub = 100
            try:
                plt.plot(np.convolve(test_loss_hist, np.ones(sub)/sub)[sub:-sub])
                filename = '%stest_loss.jpg' % hyper['graph_prefix']
                logging.info('Saving figure to: %s' % filename)
                plt.savefig(filename)
            except:
                pass

        if i % hyper['test_interval'] == 0:
            test_net.copy_params_from(net)
            for j in range(hyper['test_iter']):
                test_loss_hist.append(forward(test_net, h5filetest.next(), hyper, False)[0])
                test_net.reset_forward()
            logging.info('Validation mean error: %s' % (np.mean(test_loss_hist[-hyper['test_iter']:])))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--loglevel', default=3, type=int)
    parser.add_argument('--start_iter', default=0, type=int)
    parser.add_argument('--hyper', type=str)
    parser.add_argument('--job_num', type=int)
    parser.add_argument('--json_log_file', type=str)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--use_pal', default=False, action='store_true')
    args = parser.parse_args()
    config = imp.load_source('module.name', args.hyper)
    hyper = config.get_hyper(args.job_num)
    assert hyper['weights']
    if args.weights is not None:
        hyper['weights'] = args.weights
    hyper['use_pal'] = args.use_pal
    random.seed(0)
    apollocaffe.Caffe.set_random_seed(hyper['random_seed'])
    apollocaffe.Caffe.set_mode_gpu()
    apollocaffe.Caffe.set_device(args.gpu)
    apollocaffe.Caffe.set_logging_verbosity(args.loglevel)
    hyper['start_iter'] = args.start_iter
    with open(args.json_log_file, 'w') as f:
        json.dump(hyper, f, indent=4)

    train(hyper)

if __name__ == '__main__':
    main()
