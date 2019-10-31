import numpy as np
import logging
import nibabel as nib
import math
import argparse
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
    
    data_dir = Path(config.preprocess.data_dir)
    output_dir = Path(config.preprocess.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    Z_CROP_SIZE = int(config.preprocess.z_crop_size)
    N_SEGMENTS = int(config.preprocess.n_segments)
    COMPACTNESS = float(config.preprocess.compactness)
    N_VERTEXS = int(config.preprocess.n_vertexs)
    N_FEATURES = int(config.preprocess.n_features)
    TAO = float(config.preprocess.tao)
    # THRESHOLD = float(config.preprocess.threshold)

    paths = [path for path in data_dir.iterdir() if path.is_dir()]
    
    for path in paths:
        logging.info(f'Process {path.parts[-1]}.')
        # Create output directory
        if not (output_dir / path.parts[-1]).is_dir():
            (output_dir / path.parts[-1]).mkdir(parents=True)

    

        # Read in the CT scans
        image = nib.load(str(path / 'imaging.nii.gz')).get_data().astype(np.float32)
        label = nib.load(str(path / 'segmentation.nii.gz')).get_data().astype(np.uint8)
        image = np.transpose(image, (1, 2, 0)) # (D, W, H) -> (W, H, D)
        label = np.transpose(label, (1, 2, 0))

        image, label = zaxis_crop(image, label, Z_CROP_SIZE)
        # label_only_tumor = label.copy()
        # label_only_tumor[np.where(label==1)] = 0
        # clf_label = np.array(1) if np.count_nonzero(label_only_tumor) > 0 else np.array(0)

        image = max_min_normalize(image)
        segments = slic(image.astype('double'), n_segments=N_SEGMENTS, compactness=COMPACTNESS, multichannel=False)
        features = feature_extract(image, segments, N_VERTEXS, N_FEATURES)
        adj_arr = adj_generate(features, segments, TAO)
        gcn_label = label_transform(label, segments, N_VERTEXS, N_FEATURES)
        
        # np.save(str(output_dir / path.parts[-1] / f'image.npy'), image)
        # np.save(str(output_dir / path.parts[-1] / f'label.npy'), label)
        # np.save(str(output_dir / path.parts[-1] / f'clf_label.npy'), clf_label)
        np.save(str(output_dir / path.parts[-1] / f'segments.npy'), segments)
        np.save(str(output_dir / path.parts[-1] / f'features.npy'), features)
        np.save(str(output_dir / path.parts[-1] / f'adj_arr.npy'), adj_arr)
        np.save(str(output_dir / path.parts[-1] / f'gcn_label.npy'), gcn_label)
#         if image.shape[0]!=512 or image.shape[1]!=512:
#             print(f'woops {image.shape}')

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


def max_min_normalize(img):
    img = np.asarray(img)
    return (img-np.min(img)) / ((np.max(img)-np.min(img) + 1e-10))


def value_count(arr, num):
    """
    Args:
        arr: value 0~1 (numpy.ndarray)
    Returns:
        arr of value count: (numpy.ndarray)
    """
    arr = arr.flatten()
    arr = (arr * (num-1)).astype(np.int32)
    cnt_arr = np.zeros(num).astype(np.int32)
    for i in range(arr.shape[0]):
        cnt_arr[arr[i]] += 1
    return cnt_arr / (cnt_arr.max() + 1e-10)


def feature_extract(img, segments, v_num, f_num):
    features = np.zeros((v_num, f_num))
    for i in tqdm(range(v_num)):
        area = img[np.where(segments==i)]
        features[i] = value_count(area, f_num)
    return features 


def adj_generate(features, segments, tao):
    centroid = []
    range_num = features.shape[0] if features.shape[0] < (segments.max()+1) else (segments.max()+1)
    for i in range(range_num):
        centroid.append((np.where(segments==i)[0].mean(), np.where(segments==i)[1].mean()))
    centroid = np.asarray(centroid)
    adj_arr = np.zeros((features.shape[0], features.shape[0]))
    for i in tqdm(range(range_num)):
        for j in range(i+1, range_num):
            # if LA.norm(centroid[i] - centroid[j]) > (segments.shape[0] * threshold):
            #     adj_arr[i, j] = 0
            # else:
            #     e_dist = LA.norm((features[i] - features[j]))
            #     tmp = -1 * e_dist * e_dist / (2*tao*tao)
            #     adj_arr[i, j] = math.exp(tmp)
            e_dist = LA.norm((features[i] - features[j]))
            tmp = -1 * e_dist * e_dist / (2*tao*tao)
            adj_arr[i, j] = math.exp(tmp)
            adj_arr[j, i] = adj_arr[i, j]
    adj_arr += np.eye(features.shape[0])
    return adj_arr


def label_transform(label, segments, v_num, f_num):
    gcn_label = np.zeros((v_num, 1))
    n_range = int(np.minimum(v_num, segments.max()+1))
    for i in range(n_range):
        gcn_label[i] = int(np.median(label[np.where(segments==i)]))
    return gcn_label


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
