import numpy as np
import logging
import torch
import math
import argparse
import dgl
from dgl.data.utils import save_graphs
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.segmentation import slic
from pathlib import Path
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from tqdm import tqdm
from box import Box

def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    
    data_dir = Path(config.preprocess.output_dir)
    output_dir = Path(config.preprocess.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    paths = [path for path in data_dir.iterdir() if path.is_dir()]
    
    for path in paths:
        logging.info(f'Process {path.parts[-1]}.')
        # Create output directory
        if not (output_dir / path.parts[-1]).is_dir():
            (output_dir / path.parts[-1]).mkdir(parents=True)

        feature = np.load(path / f'features.npy').astype(np.float64) 
        adj_arr = np.load(path / f'adj_arr.npy').astype(np.float64) 
        label = np.load(path / f'clf_label.npy').astype(np.int32) 
        feature = torch.FloatTensor(feature)
        adj_arr = torch.FloatTensor(adj_arr)
        label = torch.LongTensor(label)

        n_node = feature.size(0)
        g = dgl.DGLGraph()
        g.add_nodes(n_node)
        for i in tqdm(range(n_node)):
            src = list(range(n_node))
            dst = [i] * n_node
            g.add_edges(src, dst)
            val = torch.FloatTensor(n_node, 2)
            val[:, 0] = adj_arr[i]
            val[:, 1] = adj_arr[i]
            g.edges[src, dst].data['w'] = val
        g.ndata['x'] = feature
        g_list = []
        g_list.append(g)

        save_graphs(str(path / f'graph.bin'), g_list)


def zaxis_crop(img, label, crop_size):
    """
    Args:
        img: (numpy.ndarray) (h, w, d)
    Returns:
        cropped_img: (numpy.ndarray)
    """
    d_size = img.shape[2]
    d_line = img.sum(tuple((0, 1)))

    pos_list = []
    for i in range(d_size):
        if d_line[i]!=0:
            pos_list.append(i)
    if len(pos_list) == 0:
        d_central = int(d_size / 2)
    else:
        d_central = int(sum(pos_list) / len(pos_list))

    d_front = d_central - int(crop_size / 2)
    d_front = d_front if d_front > 0 else 0
    d_back = d_front + crop_size

    return img[:, :, d_front:d_back], label[:, :, d_front:d_back]





def parse_args():
    parser = argparse.ArgumentParser(description="The script for the training and the testing.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_args()
    main(args)
