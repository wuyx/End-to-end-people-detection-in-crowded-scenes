import os
import cv2
import re
import sys
import argparse
import numpy as np
import copy
import annolist.AnnotationLib as al

from scipy.misc import imread, imresize, imsave
from munkres import Munkres, print_matrix, make_cost_matrix

class Rect(object):
    def __init__(self, cx, cy, width, height, confidence):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.confidence = confidence
        self.true_confidence = confidence
    def overlaps(self, other):
        if abs(self.cx - other.cx) > (self.width + other.width) / 2:
            return False
        elif abs(self.cy - other.cy) > (self.height + other.height) / 2:
            return False
        else:
            return True
    def distance(self, other):
        return sum(map(abs, [self.cx - other.cx, self.cy - other.cy,
                       self.width - other.width, self.height - other.height]))
    def intersection(self, other):
        left = max(self.cx - self.width/2., other.cx - other.width/2.)
        right = min(self.cx + self.width/2., other.cx + other.width/2.)
        width = max(right - left, 0)
        top = max(self.cy - self.height/2., other.cy - other.height/2.)
        bottom = min(self.cy + self.height/2., other.cy + other.height/2.)
        height = max(bottom - top, 0)
        return width * height
    def area(self):
        return self.height * self.width
    def union(self, other):
        return self.area() + other.area() - self.intersection(other)
    def iou(self, other):
        return self.intersection(other) / self.union(other)
    def __eq__(self, other):
        return (self.cx == other.cx and 
            self.cy == other.cy and
            self.width == other.width and
            self.height == other.height and
            self.confidence == other.confidence)

def filter_rects(all_rects, threshold, input_rects=[], max_threshold=1.0, config=None):
    """Takes in all_rects and based on the threshold carries out the stitching process
    as described in the paper."""

    accepted_rects = input_rects

    for i in range(0, config["grid_height"], 1):
        for j in range(0, config["grid_width"], 1):
            relevant_rects = []
            current_rects = [r for r in all_rects[i][j] if r.confidence > threshold]

            for other in accepted_rects:
                for current in current_rects:
                    if other.overlaps(current):
                        relevant_rects.append(other)
                        break

            if len(relevant_rects) == 0 or len(current_rects) == 0:
                accepted_rects += current_rects
                continue

            matrix = []
            for c in current_rects:
                row = []
                for a in relevant_rects:
                    row.append(1000)
                    if a.overlaps(c):
                        row[-1] -=100
                    row[-1] += a.distance(c) / 1000.0
                matrix.append(row)

            m = Munkres()
            cost_matrix = make_cost_matrix(matrix, lambda x: x)
            indices = m.compute(matrix)

            bad = set()
            for row, column in indices:
                c = current_rects[row]
                a = relevant_rects[column]
                if c.confidence > max_threshold:
                    bad.add(row)
                    continue
                if c.overlaps(a):
                    if c.confidence > a.confidence and c.iou(a) > 0.7:
                        c.true_confidence = a.confidence
                        accepted_rects.remove(a)
                    else:
                        bad.add(row)

            for k in range(len(current_rects)):
                if k not in bad:
                    accepted_rects.append(current_rects[k])

    return accepted_rects

def load_data_mean(data_mean_filename, img_width, img_height, image_scaling = 1.0):
    data_mean = np.load(data_mean_filename)
    data_mean = data_mean.astype(np.float32) / image_scaling
    data_mean = np.transpose(data_mean, (1, 2, 0))

    data_mean = imresize(data_mean, size=(img_height, img_width), interp='bicubic')

    data_mean = data_mean.astype(np.float32) / image_scaling
    return data_mean

def image_to_h5(I, data_mean, image_scaling = 1.0):

    # normalization as needed for ipython notebook
    I = I.astype(np.float32) / image_scaling - data_mean

    # MA: model expects BGR ordering
    I = I[:, :, (2, 1, 0)]

    data_shape = (1, I.shape[2], I.shape[0], I.shape[1])
    h5_image = np.transpose(I, (2,0,1)).reshape(data_shape) 
    return h5_image

def annotation_to_h5(a, cell_width, cell_height, region_size, max_len):
    cell_regions = get_cell_grid(cell_width, cell_height, region_size)

    cells_per_image = len(cell_regions)

    box_list = [[] for idx in range(cells_per_image)]
            
    for cidx, c in enumerate(cell_regions):
        box_list[cidx] = [r for r in a.rects if all(r.intersection(c))]

    boxes = np.zeros((1, cells_per_image, 4, max_len, 1), dtype = np.float)
    box_flags = np.zeros((1, cells_per_image, 1, max_len, 1), dtype = np.float)

    for cidx in xrange(cells_per_image):
        cur_num_boxes = min(len(box_list[cidx]), max_len)
        #assert(cur_num_boxes <= max_len)

        box_flags[0, cidx, 0, 0:cur_num_boxes, 0] = 1

        cell_ox = 0.5*(cell_regions[cidx].x1 + cell_regions[cidx].x2)
        cell_oy = 0.5*(cell_regions[cidx].y1 + cell_regions[cidx].y2)

        for bidx in xrange(cur_num_boxes):

            # relative box position with respect to cell
            ox = 0.5 * (box_list[cidx][bidx].x1 + box_list[cidx][bidx].x2) - cell_ox
            oy = 0.5 * (box_list[cidx][bidx].y1 + box_list[cidx][bidx].y2) - cell_oy

            width = abs(box_list[cidx][bidx].x2 - box_list[cidx][bidx].x1)
            height= abs(box_list[cidx][bidx].y2 - box_list[cidx][bidx].y1)

            boxes[0, cidx, :, bidx, 0] = np.array([ox, oy, width, height], dtype=np.float)

    return boxes, box_flags

def get_cell_grid(cell_width, cell_height, region_size):

    cell_regions = []
    cell_size = 32
    for iy in xrange(cell_height):
        for ix in xrange(cell_width):
            cidx = iy*cell_width + ix

            ox = (ix + 0.5)*cell_size
            oy = (iy + 0.5)*cell_size

            r = al.AnnoRect(ox - 0.5*region_size, oy - 0.5*region_size, 
                    ox + 0.5*region_size, oy + 0.5*region_size)
            r.track_id = cidx

            cell_regions.append(r)


    return cell_regions

def annotation_jitter(a_in, min_box_width=20, jitter_scale_min=0.9, jitter_scale_max=1.1, jitter_offset=16, target_width=640, target_height=480):
    a = copy.deepcopy(a_in)

    # MA: sanity check
    for r in a:
        assert(r.x1 < r.x2 and r.y1 < r.y2)

    if a.rects:
        cur_min_box_width = min([r.width() for r in a.rects])
    else:
        cur_min_box_width = min_box_width / jitter_scale_min

    # don't downscale below min_box_width 
    jitter_scale_min = max(jitter_scale_min, float(min_box_width) / cur_min_box_width)

    # it's always ok to upscale 
    jitter_scale_min = min(jitter_scale_min, 1.0)

    jitter_scale_max = jitter_scale_max

    jitter_scale = np.random.uniform(jitter_scale_min, jitter_scale_max)

    jitter_flip = np.random.random_integers(0, 1)

    I = imread(a.imageName)

    if jitter_flip == 1:
        I = np.fliplr(I)

        for r in a:
            r.x1 = I.shape[1] - r.x1
            r.x2 = I.shape[1] - r.x2
            r.x1, r.x2 = r.x2, r.x1

            for p in r.point:
                p.x = I.shape[1] - p.x

    I1 = cv2.resize(I, None, fx=jitter_scale, fy=jitter_scale, interpolation = cv2.INTER_CUBIC)

    jitter_offset_x = np.random.random_integers(-jitter_offset, jitter_offset)
    jitter_offset_y = np.random.random_integers(-jitter_offset, jitter_offset)



    rescaled_width = I1.shape[1]
    rescaled_height = I1.shape[0]

    px = round(0.5*(target_width)) - round(0.5*(rescaled_width)) + jitter_offset_x
    py = round(0.5*(target_height)) - round(0.5*(rescaled_height)) + jitter_offset_y

    I2 = np.zeros((target_height, target_width, 3), dtype=I1.dtype)

    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(rescaled_width, target_width - x1)
    y2 = min(rescaled_height, target_height - y1)

    I2[0:(y2 - y1), 0:(x2 - x1), :] = I1[y1:y2, x1:x2, :]

    ox1 = round(0.5*rescaled_width) + jitter_offset_x
    oy1 = round(0.5*rescaled_height) + jitter_offset_y

    ox2 = round(0.5*target_width)
    oy2 = round(0.5*target_height)

    for r in a:
        r.x1 = round(jitter_scale*r.x1 - x1)
        r.x2 = round(jitter_scale*r.x2 - x1)

        r.y1 = round(jitter_scale*r.y1 - y1)
        r.y2 = round(jitter_scale*r.y2 - y1)

        if r.x1 < 0:
            r.x1 = 0

        if r.y1 < 0:
            r.y1 = 0

        if r.x2 >= I2.shape[1]:
            r.x2 = I2.shape[1] - 1

        if r.y2 >= I2.shape[0]:
            r.y2 = I2.shape[0] - 1

        for p in r.point:
            p.x = round(jitter_scale*p.x - x1)
            p.y = round(jitter_scale*p.y - y1)

        # MA: make sure all points are inside the image
        r.point = [p for p in r.point if p.x >=0 and p.y >=0 and p.x < I2.shape[1] and p.y < I2.shape[0]]

    new_rects = []
    for r in a.rects:
        if r.x1 <= r.x2 and r.y1 <= r.y2:
            new_rects.append(r)
        else:
            pass

    a.rects = new_rects

    return I2, a

