#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# By: Gianni Galbiati

# Standard libraries

# External libraries
import torch
import torch.nn as nn

# Internal libraries


# NMS


def min_nms(boxes, scores, iou_threshold=0.5, mode='min'):
    """Return subset of boxes with maximal coverage using NMS algorithm.

    Imitates `torchvision.ops.nms` interface.

    NB: torchvision.ops.nms gives different results.
        Not sure why.
        torchvision.ops.nms is a compiled function;
        source location and inspection is an exercise for a motivated reader.

    Arguments
    ---------
    boxes : torch.Tensor
        size [num_boxes, 4]
        Bounding box top left and bottom right coordinates.

    scores : torch.Tensor
        size [num_boxes]
        Score for each bounding box.

    Returns
    -------
    kept : torch.Tensor
        size [new_num_boxes]
        Indices for boxes to be kept.
    """

    # No boxes? Return empty tensor.
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)

    x0, y0, x1, y1 = [boxes[:, i] for i in range(4)]
    areas = (x1 - x0 + 1.0) * (y1 - y0 + 1.0)
    score_order = torch.argsort(scores)

    kept = []

    while len(score_order) > 0:
        # Select index for highest score
        order_ix = len(score_order) - 1
        score_ix = score_order[order_ix]
        kept.append(score_ix)

        # Compute intersections of highest scoring box with the rest

        # Top left - shrink to intersection
        max_x0 = torch.max(x0[score_ix], x0[score_order[:order_ix]])
        max_y0 = torch.max(y0[score_ix], y0[score_order[:order_ix]])

        # Bottom right - same
        min_x1 = torch.min(x1[score_ix], x1[score_order[:order_ix]])
        min_y1 = torch.min(y1[score_ix], y1[score_order[:order_ix]])

        width = torch.clamp(min_x1 - max_x0 + 1.0, min=0)
        height = torch.clamp(min_y1 - max_y0 + 1.0, min=0)

        intersection_area = width * height

        if mode == 'min':
            denom = torch.min(areas[score_ix], areas[score_order[:order_ix]])
        elif mode == 'union':
            # True IOU metric
            denom = areas[score_ix] + areas[score_order[:order_ix]] - intersection_area

        overlap = intersection_area / denom

        # Only compare further boxes where overlap is below cutoff
        keep = (overlap <= iou_threshold).nonzero(as_tuple=True)[0]
        score_order = score_order[keep[keep != order_ix]]

    kept = torch.stack(kept)

    return kept


def batched_nms(boxes, n, threshold, mode='union'):
    """Applies NMS in a batched fashion, only comparing boxes from same item.

    NB: there is a (pretty sneaky) batched NMS function available
        at torchvision.ops.boxes.batched_nms

        However, since there are some differences
        between OP and torchvision NMS algorithms,
        and OP version loops in python anyway,
        we should use a simple version of our own

    Arguments
    ---------
    boxes : torch.Tensor
        size [num_boxes, 10]
        Each row is a single bounding box.

        Column 0 is batch index.
        Columns 1 - 4 are bounding box top left and bottom right coordinates.
        Column 5 is score for that box.
        Columns 6-10 are offset values.

    n : int
        number of items in a batch

    threshold : float
        IOU threshold for NMS

    mode : str
        'union' | 'min'
        'union': true IOU
        'min': divide intersection by minimum of areas instead of union

    Returns
    ------
    kept : torch.Tensor
        size [num_boxes, 10]
        Each row is a single bounding box.

        Column 0 is batch index.
        Columns 1 - 4 are bounding box top left and bottom right coordinates.
        Column 5 is score for that box.
        Columns 6-10 are offset values.
    """

    kept = []

    # For each batch item
    for bi in range(n):
        # Logical selector for batch item boxes
        selector = boxes[:, 0] == bi

        # Select boxes and scores for current item
        boxes_ = boxes[selector, 1:5]
        scores_ = boxes[selector, 5]

        if mode == 'union':
            keep = min_nms(boxes_, scores_, iou_threshold=threshold)
        elif mode == 'min':
            keep = min_nms(boxes_, scores_, iou_threshold=threshold)
        else:
            raise ValueError("mode argument must be either union or min")

        # Retain selected boxes for current item
        kept.append(boxes[selector, :][keep, :])

    # Repack into original data format
    kept = torch.cat(kept, dim=0)
    return kept


# Data structuring


def compute_boxes(scores, offsets, scale, threshold):
    """Return bounding boxes, scores, offsets, and batch indices in matrix.

    PNet acts like a 12x12 convolution with stride 2,
    so need to convert bounding box indices back to original image coordinates.

    Arguments
    ---------
    scores : torch.Tensor
        size [n, 1, h, w]
        score for face presence at each image location

    offsets : torch.Tensor
        size[n, 4, h, w]
        offsets for each image location to recover full image coordinates

    scale : float
        scaling of original image prior to PNet application

    threshold : float
        minimum score value for inclusion

    Returns
    -------
    bounding_boxes : torch.Tensor
        size [num_boxes, 10]
        Each row is a single bounding box.

        Column 0 is batch index.
        Columns 1 - 4 are bounding box top left and bottom right coordinates.
        Column 5 is score for that box.
        Columns 6-10 are offset values.
    """
    stride = 2
    kernel_size = 12

    detection_indices = (scores > threshold).nonzero(as_tuple=True)
    batch_ix = detection_indices[0]

    if batch_ix.size()[0] == 0:
        return None

    offsets_ = offsets[batch_ix, :, detection_indices[1], detection_indices[2]]
    h_ix, w_ix = [stride * d + 1 for d in detection_indices[1:]]
    scores_ = scores[batch_ix, detection_indices[1], detection_indices[2]]

    bounding_boxes = torch.stack([batch_ix.to(torch.float32),
                                  torch.round(w_ix / scale),
                                  torch.round(h_ix / scale),
                                  torch.round((w_ix + kernel_size) / scale),
                                  torch.round((h_ix + kernel_size) / scale),
                                  scores_,
                                  offsets_[:, 0],
                                  offsets_[:, 1],
                                  offsets_[:, 2],
                                  offsets_[:, 3]], dim=1)

    return bounding_boxes


# Box manipulation


def boxes_to_square(bounding_boxes):
    """Convert bounding box coordinates to a square aspect ratio."""
    x1, y1, x2, y2 = [bounding_boxes[:, i] for i in range(1, 5)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = torch.max(h, w)

    bounding_boxes[:, 1] = x1 + w * 0.5 - max_side * 0.5
    bounding_boxes[:, 2] = y1 + h * 0.5 - max_side * 0.5
    bounding_boxes[:, 3] = bounding_boxes[:, 1] + max_side - 1.0
    bounding_boxes[:, 4] = bounding_boxes[:, 2] + max_side - 1.0

    return bounding_boxes


def adjust_boxes(bounding_boxes):
    """Adjust boxes by offsets."""
    x1, y1, x2, y2 = [bounding_boxes[:, i] for i in range(1, 5)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0

    w = w.unsqueeze(1)
    h = h.unsqueeze(1)

    offsets = bounding_boxes[:, 6:]

    translation = torch.cat([w, h, w, h], dim=1) * offsets
    bounding_boxes[:, 1:5] = bounding_boxes[:, 1:5] + translation

    return bounding_boxes


def prepare_crop_params(bounding_boxes, image_height, image_width):
    """Return boxes truncated by image bounds.

    Arguments
    ---------
    bounding_boxes : torch.Tensor
        size [num_boxes, 10]
        Each row is a single bounding box.

        Column 0 is batch index.
        Columns 1 - 4 are bounding box top left and bottom right coordinates.
        Column 5 is score for that box.
        Columns 6-10 are offset values.

    image_height : int
        Height of original image.

    image_width : int
        Width of original image.

    Returns
    -------
    crop_params : list
        List of torch.tensor describing crop; each has size [num_boxes].
        [x0, x1, y0, y1, batch_index]
    """

    x0, y0, x1, y1 = [bounding_boxes[:, i] for i in range(1, 5)]

    # Truncate bottom right
    ind = (x1 > image_width - 1.0).nonzero(as_tuple=True)[0]
    x1[ind] = image_width - 1.0

    ind = (y1 > image_height - 1.0).nonzero(as_tuple=True)[0]
    y1[ind] = image_height - 1.0

    # Truncate top left
    ind = (x0 < 0.0).nonzero(as_tuple=True)[0]
    x0[ind] = 0.0

    ind = (y0 < 0.0).nonzero(as_tuple=True)[0]
    y0[ind] = 0.0

    crop_params = [y0, y1, x0, x1, bounding_boxes[:, 0]]
    crop_params = [i.to(torch.long) for i in crop_params]

    return crop_params


def crop_boxes(image, bounding_boxes, size=24):
    """Return crops from image described by bounding boxes."""
    n, c, h, w = image.size()

    crops = []

    crop_params = zip(*prepare_crop_params(bounding_boxes, h, w))
    for y0, y1, x0, x1, bi in crop_params:
        crop = image[bi, :, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)
        crop = nn.functional.interpolate(crop, size=(size, size),
                                         mode='bilinear',  align_corners=False)

        crops.append(crop.squeeze(0))

    if len(crops) > 0:
        return torch.stack(crops)
    else:
        return []
