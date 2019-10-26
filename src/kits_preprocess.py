import numpy as np
import logging
import nibabel as nib
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.segmentation import slic
from pathlib import Path
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from tqdm import tqdm


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)

    data_dir = Path(config.preprocess.data_dir)
    output_dir = Path(config.preprocess.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    CROP_SIZE = int(config.preprocess.crop_size)
    N_SEGMENTS = int(config.preprocess.n_segments)
    COMPACTNESS = float(config.preprocess.compactness)
    N_VERTEXS = int(config.preprocess.n_vertexs)
    N_FEATURES = int(config.preprocess.n_features)
    TAO = float(config.preprocess.tao)
    THRESHOLD = float(config.preprocess.threshold)

    data_dir = args.data_dir
    output_dir = args.output_dir
    paths = [path for path in data_dir.iterdir() if path.is_dir()]

    for path in paths:
        logging.info(f'Process {path.parts[-1]}.')

        # Create output directory
        if not (output_dir / path.parts[-1]).is_dir():
            (output_dir / path.parts[-1]).mkdir(parents=True)

        # Read in the CT scans
        image = nib.load(str(path / 'imaging.nii.gz')).get_data().astype(np.float32)
        label = nib.load(str(path / 'segmentation.nii.gz')).get_data().astype(np.uint8)

        # Save each slice of the scan into single file
        for s in range(image.shape[0]):
            _image = image[s:s+1].transpose((1, 2, 0)) # (C, H, W) --> (H, W, C)
            nib.save(nib.Nifti1Image(_image, np.eye(4)), str(output_dir / path.parts[-1] / f'imaging_{s}.nii.gz'))

            # The label for segmentation task.
            _seg_label = label[s:s+1].transpose((1, 2, 0)) # (C, H, W) --> (H, W, C)
            nib.save(nib.Nifti1Image(_seg_label, np.eye(4)), str(output_dir / path.parts[-1] / f'segmentation_{s}.nii.gz'))

            # The label for classification task. If the slice has kidney or tumor (foreground), the label is set to 1, otherwise is 0.
            _clf_label = np.array(1) if np.count_nonzero(_seg_label) > 0 else np.array(0)
            np.save(output_dir / path.parts[-1] / f'classification_{s}.npy', _clf_label)


def various_crop(img, label, least_size):
    """
    Args:
        img: (numpy.ndarray)
    Returns:
        cropped_img: (numpy.ndarray)
    """
    h_size = img.shape[0]
    w_size = img.shape[1]
    anchor = [0, h_size, 0, w_size]
    line_w = img.sum(0)
    line_h = img.sum(1)
    flag = False
    for i in range(h_size):
        if flag==False and line_h[i]!=0:
            anchor[0] = i
            flag = True
        elif flag==True and line_h[i]==0:
            anchor[1] = i
            flag = False
            break;
    flag = False
    for i in range(w_size):
        if flag==False and line_w[i]!=0:
            anchor[2] = i
            flag = True
        elif flag==True and line_w[i]==0:
            anchor[3] = i
            flag = False
            break;
    h_up, h_down, w_left, w_right = anchor
    if (h_down - h_up) < least_size:
        h_down = h_up + least_size
    if (w_right - w_left) < least_size:
        w_right = w_left + least_size
    return img[h_up:h_down, w_left:w_right], label[h_up:h_down, w_left:w_right]


def central_crop(img, label, crop_size):
    """
    Args:
        img: (numpy.ndarray)
    Returns:
        cropped_img: (numpy.ndarray)
    """
    h_size = img.shape[0]
    w_size = img.shape[1]
    line_h = img.sum(1)
    line_w = img.sum(0)

    pos_list = []
    for i in range(h_size):
        if line_h[i]!=0:
            pos_list.append(i)
    if len(pos_list) == 0:
        h_central = int(h_size / 2)
    else:
        h_central = int(sum(pos_list) / len(pos_list))
    pos_list = []
    for i in range(w_size):
        if line_w[i]!=0:
            pos_list.append(i)
    if len(pos_list) == 0:
        w_central = int(w_size / 2)
    else:
        w_central = int(sum(pos_list) / len(pos_list))

        h_up = h_central - int(crop_size / 2)
        h_up = h_up if h_up > 0 else 0
        w_left = w_central - int(crop_size / 2)
        w_left = w_left if w_left > 0 else 0
        h_down = h_up + crop_size
        w_right = w_left + crop_size

    return img[h_up:h_down, w_left:w_right], label[h_up:h_down, w_left:w_right]


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
    for i in range(v_num):
        area = img[np.where(segments==i)]
        features[i] = value_count(area, f_num)
    return features 


def adj_generate(features, segments, tao, threshold):
    centroid = []
    range_num = features.shape[0] if features.shape[0] < (segments.max()+1) else (segments.max()+1)
    for i in range(range_num):
        centroid.append((np.where(segments==i)[0].mean(), np.where(segments==i)[1].mean()))
    centroid = np.asarray(centroid)
    adj_arr = np.zeros((features.shape[0], features.shape[0]))
    for i in range(range_num):
        for j in range(i+1, range_num):
            if LA.norm(centroid[i] - centroid[j]) > (segments.shape[0]*threshold):
                adj_arr[i, j] = 0
            else:
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
    args = parse_args()
    main(args)
