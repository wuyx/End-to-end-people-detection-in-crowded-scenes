import os
import cv2
import re
import sys
import argparse
from annotation.annolist.python import AnnotationLib as al
import numpy as np;
import copy;

from scipy.misc import imread, imresize, imsave;

from annolist_resize_pad import resize_pad_image_helper;


# MA: minimal height after jitter
#min_box_width = 14;

# def get_keypoint_pairs():
#     keypoint_names = ['Nose',
#                       'L_Shoulder', 'L_Elbow', 'L_Wrist', 
#                       'R_Shoulder', 'R_Elbow', 'R_Wrist',
#                       'L_Hip', 'L_Knee', 'L_Ankle', 
#                       'R_Hip',  'R_Knee', 'R_Ankle'];

#     keypoint_pairs = [];

#     for idx1 in xrange(0,len(keypoint_names)):
#         k1 = keypoint_names[idx1];
#         for idx2 in xrange(idx1+1,len(keypoint_names)):
#             k2 = keypoint_names[idx2];
#             if k1[1:] == k2[1:]:
#                 print 'pair: ', k1, ' - ', k2
#                 keypoint_pairs.append((idx1, idx2));

#     return keypoint_pairs;

def annotation_jitter(a_in, min_box_width=20, jitter_scale_min=0.9, jitter_scale_max=1.1, jitter_offset=16, target_width=640, target_height=480):
    a = copy.deepcopy(a_in);

    # MA: sanity check
    for r in a:
        assert(r.x1 < r.x2 and r.y1 < r.y2);

    if a.rects:
        cur_min_box_width = min([r.width() for r in a.rects]);
    else:
        cur_min_box_width = min_box_width / jitter_scale_min;

    # don't downscale below min_box_width 
    jitter_scale_min = max(jitter_scale_min, float(min_box_width) / cur_min_box_width)

    # it's always ok to upscale 
    jitter_scale_min = min(jitter_scale_min, 1.0);

    jitter_scale_max = jitter_scale_max;

    jitter_scale = np.random.uniform(jitter_scale_min, jitter_scale_max);

    jitter_flip = np.random.random_integers(0, 1);

    I = imread(a.imageName);

    if jitter_flip == 1:
        I = np.fliplr(I);

        for r in a:
            r.x1 = I.shape[1] - r.x1;
            r.x2 = I.shape[1] - r.x2;
            r.x1, r.x2 = r.x2, r.x1

            for p in r.point:
                p.x = I.shape[1] - p.x

            # flip keypoints on left/right limbs 
            # for kp in keypoint_pairs:
            #     for p in r.point:
            #         if p.id == kp[0]:
            #             p.id = kp[1];
            #         elif p.id == kp[1]:
            #             p.id = kp[0];

    #I1 = imresize(I, jitter_scale, interp='bicubic');
    I1 = cv2.resize(I, (int(jitter_scale*I.shape[0]), int(jitter_scale*I.shape[1])), interpolation = cv2.INTER_CUBIC);

    jitter_offset_x = np.random.random_integers(-jitter_offset, jitter_offset);
    jitter_offset_y = np.random.random_integers(-jitter_offset, jitter_offset);

    # print "aidx: {}, scale range: ({}, {}), scale: {}, offset: ({}, {})".format(aidx, jitter_scale_min, jitter_scale_max, 
    #                                                                             jitter_scale, jitter_offset_x, jitter_offset_y);


    rescaled_width = I1.shape[1];
    rescaled_height = I1.shape[0];

    px = round(0.5*(target_width)) - round(0.5*(rescaled_width)) + jitter_offset_x
    py = round(0.5*(target_height)) - round(0.5*(rescaled_height)) + jitter_offset_y

    I2 = np.zeros((target_height, target_width, 3), dtype=I1.dtype);

    x1 = max(0, px);
    y1 = max(0, py);
    x2 = min(rescaled_width, target_width - x1);
    y2 = min(rescaled_height, target_height - y1);

    I2[0:(y2 - y1), 0:(x2 - x1), :] = I1[y1:y2, x1:x2, :];

    ox1 = round(0.5*rescaled_width) + jitter_offset_x;
    oy1 = round(0.5*rescaled_height) + jitter_offset_y;

    ox2 = round(0.5*target_width);
    oy2 = round(0.5*target_height);

    for r in a:
        r.x1 = round(jitter_scale*r.x1 - x1);
        r.x2 = round(jitter_scale*r.x2 - x1);

        r.y1 = round(jitter_scale*r.y1 - y1);
        r.y2 = round(jitter_scale*r.y2 - y1);

        if r.x1 < 0:
            r.x1 = 0;

        if r.y1 < 0:
            r.y1 = 0;

        if r.x2 >= I2.shape[1]:
            r.x2 = I2.shape[1] - 1;

        if r.y2 >= I2.shape[0]:
            r.y2 = I2.shape[0] - 1;

        for p in r.point:
            p.x = round(jitter_scale*p.x - x1);
            p.y = round(jitter_scale*p.y - y1);

        # MA: make sure all points are inside the image
        r.point = [p for p in r.point if p.x >=0 and p.y >=0 and p.x < I2.shape[1] and p.y < I2.shape[0]]

    new_rects = [];
    for r in a.rects:
        if r.x1 <= r.x2 and r.y1 <= r.y2:
            new_rects.append(r);
        else:
            #print "skipping rectangle: ({}, {}), ({}, {})".format(r.x1, r.x2, r.y1, r.y2);
            pass

    a.rects = new_rects;

    # check that midpoint of output coincides with (midpoint of input + jitter)
    #for cidx in xrange(3):
    #    assert(I1[oy1, ox1, cidx] == I2[oy2, ox2, cidx])

    # imsave('/deep/u/andriluka/tmp/' + str(aidx) + '_I.png', I);
    # imsave('/deep/u/andriluka/tmp/' + str(aidx) + '_I2.png', I2);

    return I2, a



if __name__ == "__main__":

    parser = argparse.ArgumentParser();
    parser.add_argument("input_filename", type=str, help="input annolist files (.idl/.al/.pal)");
    parser.add_argument("-o", "--output_dir", type=str, help="output directory for hdf5 files");

    parser.add_argument("--jitter_scale_min",  type=float, default=0.9, help="min jitter scale");
    parser.add_argument("--jitter_scale_max",  type=float, default=1.1, help="max jitter scale");
    parser.add_argument("--jitter_offset", type=float, default=16, help="jitter offset");
    parser.add_argument("--min_box_width", type=int, default=14, help="min size of the box after jitter");

    parser.add_argument("--jitter_idx", type=int, default=-1, help="output dataset index");

    # TODO: add rotation 

    args = parser.parse_args()
    annolist = al.parse(args.input_filename);

    jitter_set_idx = 0;
    output_dir = '';

    if args.jitter_idx == -1:
        while jitter_set_idx == 0 or os.path.exists(output_dir):
            jitter_set_idx += 1;
        output_dir = args.output_dir + "/jitter_" + os.path.splitext(os.path.basename(args.input_filename))[0] + "_" + str(jitter_set_idx);

        print "creating ", output_dir;
        os.makedirs(output_dir);
    else:
        jitter_set_idx = args.jitter_idx;
        output_dir = args.output_dir + "/jitter_" + os.path.splitext(os.path.basename(args.input_filename))[0] + "_" + str(jitter_set_idx);
        
        if os.path.isdir(output_dir):
            print "\nERROR: {} already exists\n".format(output_dir)
            sys.exit(0)

        os.makedirs(output_dir);
               

    np.random.seed(42 + jitter_set_idx);

    num_rects = sum(1 for a in annolist for r in a.rects);

    # MA: special for our pascal layout
    #keypoint_pairs = get_keypoint_pairs();

    annolist_out = al.AnnoList([]);

    for aidx, a_in in enumerate(annolist):
        I2, a = annotation_jitter(a_in, min_box_width=args.min_box_width, jitter_scale_min = args.jitter_scale_min, jitter_scale_max = args.jitter_scale_max, jitter_offset = args.jitter_offset);

        annolist[aidx].imageName = annolist[aidx].imageName.replace("//", "/");

        image_dir = os.path.basename(os.path.dirname(annolist[aidx].imageName));
        output_filename = output_dir + "/jitter_" + image_dir + "_" + os.path.splitext(os.path.basename(annolist[aidx].imageName))[0] + ".png"
        print "output_filename: ", output_filename;
        
        imsave(output_filename, I2);
        a.imageName = output_filename;
        annolist_out.append(a);
        
    num_rects_new = sum(1 for a in annolist for r in a.rects);

    print "num_rects before:", num_rects;
    print "num_rects after:", num_rects_new;
        
    output_annolist_filename = output_dir + "/jitter_" + os.path.basename(args.input_filename);
    annolist_out.save(output_annolist_filename);
