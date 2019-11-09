import csv
import glob
import torch
import numpy as np
import nibabel as nib
import dgl
from tqdm import tqdm

from src.data.datasets.base_dataset import BaseDataset
from src.data.transforms import compose
from dgl.data.utils import load_graphs


class DGLClfDataset(BaseDataset):
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
                    graph_paths = sorted(list((self.data_dir / case_name).glob('graph.bin')))
                   

                    self.data_paths.extend([(graph_path, label_path) for graph_path, label_path in zip(graph_paths, label_paths)])


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        graph_path, label_path = self.data_paths[index]
        
        g_list, label_dict = load_graphs(str(graph_path))
        graph = g_list[0]

        label = np.expand_dims(np.load(label_path).astype(np.int64), 0)
        #label = np.expand_dims(label, 1)
        label = self.transforms(label, dtypes=[torch.long]) 
        


        label = label.contiguous()
        # print(graph_path)
        # return {"graph": graph, "label": label}
        return [graph, label]

