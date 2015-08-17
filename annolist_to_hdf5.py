from annotation.annolist.python import AnnotationLib as al

import os;
import sys;

import argparse;
import h5py
import numpy as np;

import pickle as pkl;

from scipy.misc import imread, imresize;
from annolist_resize_pad import resize_pad_image_helper

# MA: use imread from opencv to be compatible with "compute_driving_mean.cpp -> convert_mean.py" pipeline
import cv2;

def get_trailing_num(str):
    res = re.search("\d*$", str);

    if res.start() < len(str):
        n = int(str[res.start():]);
    else:
        return None;


# generate labels in hdf5 format
# "image", (1, 3, h, w)
# "boxes", (300, 1, 20, 4)
# "box_flags", (300, 1, 20, 1)


# cells_per_image = 300;
# cell_width = 20;
# cell_height = 15;
#cell_size = 640 / cell_width;
cell_size = 32

# MA: max number of boxes to predict
#boxes_per_cell = 20;
boxes_per_cell = 5;
#boxes_per_cell = 4;
#boxes_per_cell = 7;

# MA: region used to collecting boxes for prediction
#region_size = 32;
region_size = 64;
#region_size = 96;

#def get_cell_grid():
def get_cell_grid(cell_width, cell_height):
    # global cells_per_image;
    # global cell_width;
    # global cell_height;

    cell_regions = [];
    for iy in xrange(cell_height):
        for ix in xrange(cell_width):
            cidx = iy*cell_width + ix;

            ox = (ix + 0.5)*cell_size;
            oy = (iy + 0.5)*cell_size;

            r = al.AnnoRect(ox - 0.5*region_size, oy - 0.5*region_size, ox + 0.5*region_size, oy + 0.5*region_size);
            r.track_id = cidx;

            cell_regions.append(r);


    return cell_regions;

def load_data_mean(data_mean_filename, img_width, img_height, image_scaling = 1.0):
    data_mean = np.load(data_mean_filename);
    data_mean = data_mean.astype(np.float32) / image_scaling
    data_mean = np.transpose(data_mean, (1, 2, 0))

    #data_mean = imresize(data_mean, size=(480, 640))
    #data_mean = imresize(data_mean, size=(480, 640), interp='bicubic')
    data_mean = imresize(data_mean, size=(img_height, img_width), interp='bicubic')

    data_mean = data_mean.astype(np.float32) / image_scaling
    return data_mean;

def image_to_h5(I, data_mean, image_scaling = 1.0):

    # normalization as needed for ipython notebook
    I = I.astype(np.float32) / image_scaling - data_mean;

    # MA: model expects BGR ordering
    I = I[:, :, (2, 1, 0)];

    data_shape = (1, I.shape[2], I.shape[0], I.shape[1])
    h5_image = np.transpose(I, (2,0,1)).reshape(data_shape) 
    return h5_image;

def annotation_to_h5(a, cell_width, cell_height):
    cell_regions = get_cell_grid(cell_width, cell_height);

    cells_per_image = len(cell_regions);

    box_list = [[] for idx in range(cells_per_image)];
            
    for cidx, c in enumerate(cell_regions):
        box_list[cidx] = [r for r in a.rects if all(r.intersection(c))]

    boxes = np.zeros((1, cells_per_image, 4, boxes_per_cell, 1), dtype = np.float);
    box_flags = np.zeros((1, cells_per_image, 1, boxes_per_cell, 1), dtype = np.float);

    for cidx in xrange(cells_per_image):
        cur_num_boxes = min(len(box_list[cidx]), boxes_per_cell);
        #assert(cur_num_boxes <= boxes_per_cell);

        box_flags[0, cidx, 0, 0:cur_num_boxes, 0] = 1;

        cell_ox = 0.5*(cell_regions[cidx].x1 + cell_regions[cidx].x2);
        cell_oy = 0.5*(cell_regions[cidx].y1 + cell_regions[cidx].y2);

        for bidx in xrange(cur_num_boxes):

            # relative box position with respect to cell
            ox = 0.5 * (box_list[cidx][bidx].x1 + box_list[cidx][bidx].x2) - cell_ox;
            oy = 0.5 * (box_list[cidx][bidx].y1 + box_list[cidx][bidx].y2) - cell_oy;

            width = abs(box_list[cidx][bidx].x2 - box_list[cidx][bidx].x1);
            height= abs(box_list[cidx][bidx].y2 - box_list[cidx][bidx].y1);

            #boxes[0, cidx, 0, bidx, :] = np.array([ox, oy, width, height], dtype=np.float);
            boxes[0, cidx, :, bidx, 0] = np.array([ox, oy, width, height], dtype=np.float);

    return boxes, box_flags


def h5write(filename, image, boxes, box_flags):
    if os.path.exists(filename):
        print 
        print "file already exists: {}".format(filename)
        print 
        assert(False);

    with h5py.File(filename, 'w') as f:
         assert image.dtype == np.float32

         f.create_dataset('image', image.shape, compression='gzip')
         f['image'].write_direct(image)
         f['boxes'] = boxes;
         f['box_flags'] = box_flags;


def h5write_old(filename, image, box_list, cell_regions):
    #global cells_per_image;
    # global cell_width;
    # global cell_height;

    #assert(len(box_list) == cells_per_image);
    cells_per_image = len(box_list);

    if os.path.exists(filename):
        print 
        print "file already exists: {}".format(filename)
        print 
        assert(False);

    with h5py.File(filename, 'w') as f:
         assert image.dtype == np.float32

         # data_shape = (1, image.shape[2], image.shape[0], image.shape[1])
         # f['image'] = np.transpose(image, (2,0,1)).reshape(data_shape) 
         #f['image'] = image;
         f.create_dataset('image', image.shape, compression='gzip')
         f['image'].write_direct(image)

         #boxes = np.zeros((1, cells_per_image, 1, boxes_per_cell, 4), dtype = np.float);
         boxes = np.zeros((1, cells_per_image, 4, boxes_per_cell, 1), dtype = np.float);

         box_flags = np.zeros((1, cells_per_image, 1, boxes_per_cell, 1), dtype = np.float);

         assert(len(cell_regions) == cells_per_image)

         for cidx in xrange(cells_per_image):
             cur_num_boxes = min(len(box_list[cidx]), boxes_per_cell);
             #assert(cur_num_boxes <= boxes_per_cell);

             box_flags[0, cidx, 0, 0:cur_num_boxes, 0] = 1;

             cell_ox = 0.5*(cell_regions[cidx].x1 + cell_regions[cidx].x2);
             cell_oy = 0.5*(cell_regions[cidx].y1 + cell_regions[cidx].y2);

             for bidx in xrange(cur_num_boxes):

                 # relative box position with respect to cell
                 ox = 0.5 * (box_list[cidx][bidx].x1 + box_list[cidx][bidx].x2) - cell_ox;
                 oy = 0.5 * (box_list[cidx][bidx].y1 + box_list[cidx][bidx].y2) - cell_oy;

                 width = abs(box_list[cidx][bidx].x2 - box_list[cidx][bidx].x1);
                 height= abs(box_list[cidx][bidx].y2 - box_list[cidx][bidx].y1);

                 #boxes[0, cidx, 0, bidx, :] = np.array([ox, oy, width, height], dtype=np.float);
                 boxes[0, cidx, :, bidx, 0] = np.array([ox, oy, width, height], dtype=np.float);

         f['boxes'] = boxes;
         f['box_flags'] = box_flags;


if __name__ == "__main__":

    # global cells_per_image;
    # global cell_width;
    # global cell_height;

    parser = argparse.ArgumentParser();
    parser.add_argument("input_filename", type=str, help="input annolist files (.idl/.al/.pal)");
    parser.add_argument("-o", "--output_dir", type=str, help="output directory for hdf5 files");
    parser.add_argument("--data_mean", type=str, default="", help="dataset mean (*.npy)");
    parser.add_argument("--skip_h5", action='store_true', help="don't save h5, only generate annolist");

    parser.add_argument("--img_width", type=int, default=640, help="img_width");
    parser.add_argument("--img_height", type=int, default=480, help="img_height");
    parser.add_argument("--comp_mean_only", action='store_true', help="compute data mean and exit");

    args = parser.parse_args()

    assert(args.img_width % cell_size == 0);
    assert(args.img_height % cell_size == 0);

    cell_width = args.img_width / cell_size;
    cell_height = args.img_height / cell_size;

    # cells_per_image = cell_width * cell_height;
    # cell_regions = get_cell_grid(cell_width, cell_height);

    print "grid size: {}x{}".format(cell_height, cell_width);


    annolist = al.parse(args.input_filename);

    if not os.path.exists(args.output_dir):
        print "creating ", args.output_dir;
        os.makedirs(args.output_dir);

    image_scaling = 1.0

    # load data mean
    if not args.data_mean:
        print "computing data mean..."

        # compute data mean
        data_mean = np.zeros((args.img_height, args.img_width, 3), dtype=np.float32);

        for aidx, a in enumerate(annolist):
            print a.imageName;

            I = cv2.imread(a.imageName);

            assert(I.shape == data_mean.shape);
            data_mean += I;

        data_mean /= len(annolist);
        
        # MA: generate the same output as convert_mean.py
        
        # MA: BGR -> RGB
        data_mean = data_mean[:, :, (2, 1, 0)]

        # MA: depth channel first 
        data_mean = np.transpose(data_mean, (2, 0, 1))

        data_mean_filename = args.output_dir + "/" + os.path.splitext(os.path.basename(args.input_filename))[0] + "_data_mean.npy";
        print "saving data mean to", data_mean_filename
        np.save(open(data_mean_filename, 'wb'), data_mean)
  
    else:
        data_mean = load_data_mean(args.data_mean, args.img_width, args.img_height, image_scaling);

    if args.comp_mean_only:
        sys.exit(0);

    filenames_list = [];

    for aidx, a in enumerate(annolist):
        print "aidx: ", aidx;

        if not args.skip_h5:
            I  = imread(annolist[aidx].imageName)

            assert(I.shape[0] == args.img_height);
            assert(I.shape[1] == args.img_width);

        image_dir = os.path.basename(os.path.dirname(annolist[aidx].imageName));
        output_filename = args.output_dir + "/" + image_dir + "_" + os.path.splitext(os.path.basename(annolist[aidx].imageName))[0] + ".h5"

        filenames_list.append((os.path.abspath(output_filename), annolist[aidx].imageName));

        if not args.skip_h5:
            image = image_to_h5(I, data_mean, image_scaling)
            boxes, box_flags = annotation_to_h5(a, cell_width, cell_height)

            h5write(output_filename, image, boxes, box_flags);

    output_filename_pkl = args.output_dir + "/" + os.path.splitext(os.path.basename(args.input_filename))[0] + ".pkl"
    print "saving ", output_filename_pkl;

    with open(output_filename_pkl, 'w') as f:
        pkl.dump(filenames_list, f);
