import csv
import glob
import torch
import numpy as np
import nibabel as nib
import dgl
from tqdm import tqdm

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose

class KitsClfDataset(BaseDataset):
    """
    Args:
        data_split_csv (str): The path of the training and validation data split csv file.
        train_preprocessings (list of Box): The preprocessing techniques applied to the training data before applying the augmentation.
        valid_preprocessings (list of Box): The preprocessing techniques applied to the validation data before applying the augmentation.
        transforms (list of Box): The preprocessing techniques applied to the data.
        augments (list of Box): The augmentation techniques applied to the training data (default: None).
    """
    def __init__(self, data_split_csv, train_preprocessings, valid_preprocessings, transforms, augments=None, **kwargs):
        super().__init__(**kwargs)
        self.data_split_csv = data_split_csv
        self.train_preprocessings = compose(train_preprocessings)
        self.valid_preprocessings = compose(valid_preprocessings)
        self.transforms = compose(transforms)
        self.augments = compose(augments)
        
        self.data_paths = []
        self.input_paths = []

        # Collect the data paths according to the dataset split csv.
        with open(self.data_split_csv, "r") as f:
            type_ = 'train' if self.type == 'train' else 'valid'
            rows = csv.reader(f)
            for case_name, split_type in rows:
                if split_type == type_:
                    label_paths = sorted(list((self.data_dir / case_name).glob('clf_label*.npy')))
                    features_paths = sorted(list((self.data_dir / case_name).glob('features*.npy')))
                    adj_paths = sorted(list((self.data_dir / case_name).glob('adj_arr*.npy')))
                   

                    self.data_paths.extend([(features_path, label_path, adj_path) for features_path, label_path, adj_path in zip(features_paths, label_paths, adj_paths)])


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        feature_path, label_path, adj_path = self.data_paths[index]
        
        label = np.expand_dims(np.load(label_path).astype(np.int64), 0)
        #label = np.expand_dims(label, 1)
        feature = np.load(feature_path).astype(np.float32)
        adj_arr = np.load(adj_path).astype(np.float32)

        label, feature, adj_arr = self.transforms(label, feature, adj_arr, dtypes=[torch.long, torch.float, torch.float]) 
        
        # n_node = features.size(0)
        # g = dgl.DGLGraph()
        # g.add_nodes(n_node)
        # for i in range(n_node):
        #     src = list(range(n_node))
        #     dst = [i] * n_node
        #     g.add_edges(src, dst)
        # g.ndata['x'] = features
        # for i in tqdm(range(n_node)):
        #     src = list(range(n_node))
        #     dst = [i] * n_node
        #     val = torch.FloatTensor(n_node, 2)
        #     val[:, 0] = adj_arr[i]
        #     val[:, 1] = adj_arr[i]
        #     g.edges[src, dst].data['w'] = val 


        label, feature, adj_arr = label.contiguous(), feature.contiguous(), adj_arr.contiguous()
        
        return {"feature": feature, "label": label, "adj_arr": adj_arr}
        # return {"graph": g, "label": label}

