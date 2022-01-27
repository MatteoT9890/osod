import torch
import numpy as np
import csv
import json
import pickle
from typing import List


def top_k(value: torch.tensor, k: int, descending=True, dim=-1):
    sorted_value, sorted_indices = value.sort(descending=descending, dim=dim)
    topk_values = sorted_value.narrow(0, 0, k)
    topk_indices = sorted_indices.narrow(0, 0, k)

    return topk_values, topk_indices


"""
Compute the maximum IoU between a list of boxes and a box.
If the maximum IoU is above the threshold, it returns the corresponding index of the list of boxes; otherwise it
return -1.
"""


def get_max_iou(boxes, box, thresh: float = 0.5):
    # compute overlaps
    # intersection
    ixmin = np.maximum(boxes[:, 0], box[0])
    iymin = np.maximum(boxes[:, 1], box[1])
    ixmax = np.minimum(boxes[:, 2], box[2])
    iymax = np.minimum(boxes[:, 3], box[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
            (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
            + (boxes[:, 2] - boxes[:, 0] + 1.0) * (boxes[:, 3] - boxes[:, 1] + 1.0)
            - inters
    )

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)

    if ovmax > thresh:
        return jmax
    else:
        return -1


def swap(tensor: torch.Tensor, dim: int, idx: torch.LongTensor):
    return tensor.index_select(dim, idx)


def write_list_json_as_csv(data, header, filepath):
    with open(filepath, "w", newline="") as f:
        cw = csv.DictWriter(f, header, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cw.writeheader()
        cw.writerows(data)


def write_dict_as_json(dictionary, filepath):
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open(filepath, "w") as outfile:
        outfile.write(json_object)


def create_onehot(intLabel, num_classes, device='cuda'):
    onehot = torch.zeros(num_classes)
    onehot[intLabel] = 1
    return onehot.to(device)


def one_hot_matrix(labels, n_classes, device='cuda'):
    matrix = torch.zeros((len(labels), n_classes))
    for index, y in enumerate(labels):
        matrix[index] = create_onehot(y, n_classes, device=device)
    return matrix.to(device)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)