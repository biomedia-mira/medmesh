import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
import pyvista as pv
import pytorch_lightning as pl
import torch_geometric.transforms as T

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List
from tqdm import tqdm
from collections import defaultdict
from copy import copy

from src.transforms import MultiGraphTransform


def _subdict(): return defaultdict(str)


class UKBBDataset(Dataset):
    """
    Path should have the following directory structure:
    /path
    ├── 1000000
    │   ├── T1_first-BrStem_first.vtk
    │   ├── T1_first-L_Accu_first.vtk
    │   ├── T1_first-L_Amyg_first.vtk
    │   ├── T1_first-L_Caud_first.vtk
    │   ├── T1_first-L_Hipp_first.vtk
    │   ├── T1_first-L_Pall_first.vtk
    │   ├── T1_first-L_Puta_first.vtk
    │   ├── T1_first-L_Thal_first.vtk
    │   ├── T1_first-R_Accu_first.vtk
    │   ├── T1_first-R_Amyg_first.vtk
    │   ├── T1_first-R_Caud_first.vtk
    │   ├── T1_first-R_Hipp_first.vtk
    │   ├── T1_first-R_Pall_first.vtk
    │   ├── T1_first-R_Puta_first.vtk
    │   └── T1_first-R_Thal_first.vtk
    ├── 1000001
    │   ├── T1_first-BrStem_first.vtk
        ...
    ...

    We expect that there should be 7 meshes for the left and right side of the
    brain and another mesh for the brain stem.
    """

    def __init__(self, path: str, substructures: List[str], metadata_file: str, target_label: str, transform: T.Compose = None, reload_path: bool = False, cache_path: str = '.'):
        super().__init__(path, transform=MultiGraphTransform(transform=transform), pre_transform=None, pre_filter=None)
        self.path = path
        self.cache_path = cache_path
        self.substructures = sorted(substructures)
        self.subject_ids = []
        self.lookup_dict = defaultdict(_subdict)
        self.reload_path = reload_path

        self.subject_id_colname = 'eid'
        self.label_name_map = {
            '31-0.0': 'sex',
            '21003-2.0': 'age',
        }

        self.target_label = target_label

        cols = [self.subject_id_colname] + list(self.label_name_map.keys())

        print('Loading metadata...')
        df = pd.read_csv(metadata_file, low_memory=False)
        self.metadata_df = df[cols].rename(columns=self.label_name_map)
        self.metadata_df['dx'] = 0  # no known disease cases
        self.metadata_df.drop_duplicates(subset=[self.subject_id_colname], inplace=True)

        pickle_files = '_'.join(substructures)
        self.sub_file = f'{self.cache_path}/ukbb_subject_ids_{pickle_files}.pickle'
        self.lookup_dict_file = f'{self.cache_path}/ukbb_lookup_dict_{pickle_files}.pickle'
        self.read_path_structure()

    def read_path_structure(self):
        if not self.reload_path and os.path.exists(self.sub_file) and os.path.exists(self.lookup_dict_file):
            with open(self.sub_file, 'rb') as sub_file, \
                open(self.lookup_dict_file, 'rb') as lookup_dict_file:
                self.subject_ids = pickle.load(sub_file)
                self.lookup_dict = pickle.load(lookup_dict_file)
        else:
            all_subject_ids = sorted(
                [int(x) for x in os.listdir(self.path) if x.isdigit()]
            )

            for _id in tqdm(all_subject_ids, desc='Loading Data'):
                labels = self.lookup_labels(int(_id))
                if not labels.empty and labels['age'].notna().all() and labels['sex'].notna().all():  # exclude subject if age or sex is missing
                    valid_subject = True
                    for substructure in self.substructures:
                        full_path = self.create_vtk_path(self.path, _id, substructure)
                        if not os.path.exists(full_path):
                            valid_subject = False
                            continue
                        self.lookup_dict[_id][substructure] = full_path
                    if valid_subject:
                        self.subject_ids.append(_id)

            with open(self.sub_file, 'wb') as sub_file, \
                open(self.lookup_dict_file, 'wb') as lookup_dict_file:
                pickle.dump(self.subject_ids, sub_file)
                pickle.dump(self.lookup_dict, lookup_dict_file)

        print(f'Valid subjects: {self.len()}')

    def create_vtk_path(self, path, subject_id, substructure):
        if path[-1] == '/':
            path = path[:-1]
        return f'{path}/{subject_id}/T1_first-{substructure}_first.vtk'

    def load_mesh_file(self, full_path: str):
        return pv.read(full_path)

    def get_data_from_polydata(self, path):
        polydata = self.load_mesh_file(path)
        faces = polydata.faces.reshape(-1, 4)[:, 1:].T
        normal = polydata.point_normals
        pos = polydata.points
        data = Data(
            pos=torch.tensor(pos),
            normal=torch.tensor(normal),
            face=torch.tensor(faces),
            vtkpath = os.path.abspath(path)
        )
        return data

    def lookup_labels(self, subject_id: int):
        mask = self.metadata_df[self.subject_id_colname] == subject_id
        labels = self.metadata_df.loc[mask]
        return labels

    def len(self):
        return len(self.subject_ids)

    def get(self, idx):
        subject_id = self.subject_ids[idx]
        paths = self.lookup_dict[subject_id]
        labels = self.lookup_labels(int(subject_id))
        meshes = []
        for substructure in self.substructures:
            mesh = self.get_data_from_polydata(paths[substructure])
            meshes.append(mesh)
        sample = {}
        sample['x'] = meshes
        sample['age'] = torch.from_numpy(np.array(labels['age'], dtype='float32'))
        sample['sex'] = torch.from_numpy(np.array(labels['sex'], dtype='int64'))
        sample['dx'] = torch.from_numpy(np.array(labels['dx'], dtype='int64'))
        sample['y'] = sample[self.target_label]
        return sample

    @property
    def num_features(self) -> int:
        graph = self[0]['x'][0]
        return graph.num_node_features


class CamCANDataset(Dataset):
    def __init__(self, path: str, substructures: List[str], metadata_file: str, target_label: str, transform: T.Compose = None,
                 reload_path: bool = False, cache_path: str = '.'):
        super().__init__(path, transform=MultiGraphTransform(transform=transform), pre_transform=None, pre_filter=None)
        self.path = path
        self.cache_path = cache_path
        self.substructures = sorted(substructures)
        self.subject_ids = []
        self.lookup_dict = defaultdict(_subdict)
        self.reload_path = reload_path

        self.subject_id_colname = 'Observations'
        self.label_name_map = {
            'gender_code': 'sex',
            'age': 'age',
        }

        self.target_label = target_label

        cols = [self.subject_id_colname] + list(self.label_name_map.keys())

        print('Loading metadata...')
        df = pd.read_csv(metadata_file)
        self.metadata_df = df[cols].rename(columns=self.label_name_map)
        self.metadata_df.loc[self.metadata_df['sex'] == 2, 'sex'] = 0 # map female to label 0
        self.metadata_df['dx'] = 0  # no known disease cases
        self.metadata_df.drop_duplicates(subset=[self.subject_id_colname], inplace=True)

        pickle_files = '_'.join(substructures)
        self.sub_file = f'{self.cache_path}/camcan_subject_ids_{pickle_files}.pickle'
        self.lookup_dict_file = f'{self.cache_path}/camcan_lookup_dict_{pickle_files}.pickle'
        self.read_path_structure()

    def read_path_structure(self):
        if not self.reload_path and os.path.exists(self.sub_file) and os.path.exists(self.lookup_dict_file):
            with open(self.sub_file, 'rb') as sub_file, \
                    open(self.lookup_dict_file, 'rb') as lookup_dict_file:
                self.subject_ids = pickle.load(sub_file)
                self.lookup_dict = pickle.load(lookup_dict_file)
        else:
            for _id in tqdm(self.metadata_df[self.subject_id_colname], desc='Loading Data'):
                labels = self.lookup_labels(_id)
                if not labels.empty and labels['age'].notna().all() and labels['sex'].notna().all():  # exclude subject if age or sex is missing
                    valid_subject = True
                    for substructure in self.substructures:
                        full_path = self.create_vtk_path(self.path, _id, substructure)
                        if not os.path.exists(full_path):
                            valid_subject = False
                            continue
                        self.lookup_dict[_id][substructure] = full_path
                    if valid_subject:
                        self.subject_ids.append(_id)

            with open(self.sub_file, 'wb') as sub_file, \
                    open(self.lookup_dict_file, 'wb') as lookup_dict_file:
                pickle.dump(self.subject_ids, sub_file)
                pickle.dump(self.lookup_dict, lookup_dict_file)

        print(f'Valid subjects: {self.len()}')

    def create_vtk_path(self, path, subject_id, substructure):
        if path[-1] == '/':
            path = path[:-1]
        return f'{path}/sub-{subject_id}_T1w_unbiased-{substructure}_first.vtk'

    def load_mesh_file(self, full_path: str):
        return pv.read(full_path)

    def get_data_from_polydata(self, path):
        polydata = self.load_mesh_file(path)
        faces = polydata.faces.reshape(-1, 4)[:, 1:].T
        normal = polydata.point_normals
        pos = polydata.points
        data = Data(
            pos=torch.tensor(pos),
            normal=torch.tensor(normal),
            face=torch.tensor(faces),
            vtkpath = os.path.abspath(path)
        )
        return data

    def lookup_labels(self, subject_id):
        mask = self.metadata_df[self.subject_id_colname] == subject_id
        labels = self.metadata_df.loc[mask]
        return labels

    def len(self):
        return len(self.subject_ids)

    def get(self, idx):
        subject_id = self.subject_ids[idx]
        paths = self.lookup_dict[subject_id]
        labels = self.lookup_labels(subject_id)
        meshes = []
        for substructure in self.substructures:
            mesh = self.get_data_from_polydata(paths[substructure])
            meshes.append(mesh)
        sample = {}
        sample['x'] = meshes
        sample['age'] = torch.from_numpy(np.array(labels['age'], dtype='float32'))
        sample['sex'] = torch.from_numpy(np.array(labels['sex'], dtype='int64'))
        sample['dx'] = torch.from_numpy(np.array(labels['dx'], dtype='int64'))
        sample['y'] = sample[self.target_label]
        return sample

    @property
    def num_features(self) -> int:
        graph = self[0]['x'][0]
        return graph.num_node_features


class IXIDataset(Dataset):
    def __init__(self, path: str, substructures: List[str], metadata_file: str, target_label: str, transform: T.Compose = None,
                 reload_path: bool = False, cache_path: str = '.'):
        super().__init__(path, transform=MultiGraphTransform(transform=transform), pre_transform=None, pre_filter=None)
        self.path = path
        self.cache_path = cache_path
        self.substructures = sorted(substructures)
        self.subject_ids = []
        self.lookup_dict = defaultdict(_subdict)
        self.reload_path = reload_path

        self.subject_id_colname = 'IXI_ID'
        self.label_name_map = {
            'SEX': 'sex',
            'AGE': 'age',
        }

        self.target_label = target_label

        cols = [self.subject_id_colname] + list(self.label_name_map.keys())

        print('Loading metadata...')
        df = pd.read_csv(metadata_file)
        self.metadata_df = df[cols].rename(columns=self.label_name_map)
        self.metadata_df.loc[self.metadata_df['sex'] == 2, 'sex'] = 0 # map female to label 0
        self.metadata_df['dx'] = 0  # no known disease cases
        self.metadata_df.drop_duplicates(subset=[self.subject_id_colname], inplace=True)

        pickle_files = '_'.join(substructures)
        self.sub_file = f'{self.cache_path}/ixi_subject_ids_{pickle_files}.pickle'
        self.lookup_dict_file = f'{self.cache_path}/ixi_lookup_dict_{pickle_files}.pickle'
        self.read_path_structure()

    def read_path_structure(self):
        if not self.reload_path and os.path.exists(self.sub_file) and os.path.exists(self.lookup_dict_file):
            with open(self.sub_file, 'rb') as sub_file, \
                    open(self.lookup_dict_file, 'rb') as lookup_dict_file:
                self.subject_ids = pickle.load(sub_file)
                self.lookup_dict = pickle.load(lookup_dict_file)
        else:
            all_subject_ids = sorted(self.get_subject_ids())

            for _id in tqdm(all_subject_ids, desc='Loading Data'):
                labels = self.lookup_labels(_id)
                if not labels.empty and labels['age'].notna().all() and labels['sex'].notna().all():  # exclude subject if age or sex is missing
                    valid_subject = True
                    for substructure in self.substructures:
                        full_path = self.create_vtk_path(self.path, _id, substructure)
                        if full_path == None or not os.path.exists(full_path):
                            valid_subject = False
                            continue
                        self.lookup_dict[_id][substructure] = full_path
                    if valid_subject:
                        self.subject_ids.append(_id)


            with open(self.sub_file, 'wb') as sub_file, \
                    open(self.lookup_dict_file, 'wb') as lookup_dict_file:
                pickle.dump(self.subject_ids, sub_file)
                pickle.dump(self.lookup_dict, lookup_dict_file)

        print(f'Valid subjects: {self.len()}')

    def get_subject_ids(self):
        files = glob.glob(os.path.join(self.path, f'IXI*{self.substructures[0]}*.vtk'))
        ids = [int(os.path.basename(f).split('-')[0].replace('IXI','')) for f in files]
        return ids

    def create_vtk_path(self, path, subject_id, substructure):
        if path[-1] == '/':
            path = path[:-1]
        files = glob.glob(os.path.join(path, f'IXI{str(subject_id).zfill(3)}*{substructure}*.vtk'))
        if len(files) == 1:
            return files[0]
        else:
            return None

    def load_mesh_file(self, full_path: str):
        return pv.read(full_path)

    def get_data_from_polydata(self, path):
        polydata = self.load_mesh_file(path)
        faces = polydata.faces.reshape(-1, 4)[:, 1:].T
        normal = polydata.point_normals
        pos = polydata.points
        data = Data(
            pos=torch.tensor(pos),
            normal=torch.tensor(normal),
            face=torch.tensor(faces),
            vtkpath = os.path.abspath(path)
        )
        return data

    def lookup_labels(self, subject_id):
        mask = self.metadata_df[self.subject_id_colname] == subject_id
        labels = self.metadata_df.loc[mask]
        return labels

    def len(self):
        return len(self.subject_ids)

    def get(self, idx):
        subject_id = self.subject_ids[idx]
        paths = self.lookup_dict[subject_id]
        labels = self.lookup_labels(subject_id)
        meshes = []
        for substructure in self.substructures:
            mesh = self.get_data_from_polydata(paths[substructure])
            meshes.append(mesh)
        sample = {}
        sample['x'] = meshes
        sample['age'] = torch.from_numpy(np.array(labels['age'], dtype='float32'))
        sample['sex'] = torch.from_numpy(np.array(labels['sex'], dtype='int64'))
        sample['dx'] = torch.from_numpy(np.array(labels['dx'], dtype='int64'))
        sample['y'] = sample[self.target_label]
        return sample

    @property
    def num_features(self) -> int:
        graph = self[0]['x'][0]
        return graph.num_node_features


class OASIS3Dataset(Dataset):
    def __init__(self, path: str, substructures: List[str], metadata_file: str, target_label: str, transform: T.Compose = None,
                 reload_path: bool = False, cache_path: str = '.'):
        super().__init__(path, transform=MultiGraphTransform(transform=transform), pre_transform=None, pre_filter=None)
        self.path = path
        self.cache_path = cache_path
        self.substructures = sorted(substructures)
        self.subject_ids = []
        self.lookup_dict = defaultdict(_subdict)
        self.reload_path = reload_path

        self.subject_id_colname = 'MR ID_MR'
        self.label_name_map = {
            'M/F_MR': 'sex',
            'Age_MR': 'age',
            'cdr': 'dx',
        }

        self.target_label = target_label

        cols = [self.subject_id_colname] + list(self.label_name_map.keys())

        print('Loading metadata...')
        df = pd.read_csv(metadata_file)
        self.metadata_df = df[cols].rename(columns=self.label_name_map)
        self.metadata_df.loc[self.metadata_df['sex'] == 'F', 'sex'] = 0 # map female to label 0
        self.metadata_df.loc[self.metadata_df['sex'] == 'M', 'sex'] = 1  # map male to label 1
        self.metadata_df.loc[self.metadata_df['dx'] > 0, 'dx'] = 1  # map disease to label 1
        self.metadata_df.drop_duplicates(subset=[self.subject_id_colname], inplace=True)

        pickle_files = '_'.join(substructures)
        self.sub_file = f'{self.cache_path}/oasis3_subject_ids_{pickle_files}.pickle'
        self.lookup_dict_file = f'{self.cache_path}/oasis3_lookup_dict_{pickle_files}.pickle'
        self.read_path_structure()

    def read_path_structure(self):
        if not self.reload_path and os.path.exists(self.sub_file) and os.path.exists(self.lookup_dict_file):
            with open(self.sub_file, 'rb') as sub_file, \
                    open(self.lookup_dict_file, 'rb') as lookup_dict_file:
                self.subject_ids = pickle.load(sub_file)
                self.lookup_dict = pickle.load(lookup_dict_file)
        else:
            for _id in tqdm(self.metadata_df[self.subject_id_colname], desc='Loading Data'):
                labels = self.lookup_labels(_id)
                if not labels.empty and labels['age'].notna().all() and labels['sex'].notna().all():  # exclude subject if age or sex is missing
                    valid_subject = True
                    for substructure in self.substructures:
                        full_path = self.create_vtk_path(self.path, _id, substructure)
                        if full_path == None or not os.path.exists(full_path):
                            valid_subject = False
                            continue
                        self.lookup_dict[_id][substructure] = full_path
                    if valid_subject:
                        self.subject_ids.append(_id)

            with open(self.sub_file, 'wb') as sub_file, \
                    open(self.lookup_dict_file, 'wb') as lookup_dict_file:
                pickle.dump(self.subject_ids, sub_file)
                pickle.dump(self.lookup_dict, lookup_dict_file)

        print(f'Valid subjects: {self.len()}')

    def create_vtk_path(self, path, subject_id, substructure):
        if path[-1] == '/':
            path = path[:-1]
        parts = subject_id.split('_')
        files = glob.glob(os.path.join(path, f'sub-{str(parts[0])}*{str(parts[2])}*{substructure}*.vtk'))
        if len(files) == 1:
            return files[0]
        else:
            return None

    def load_mesh_file(self, full_path: str):
        return pv.read(full_path)

    def get_data_from_polydata(self, path):
        polydata = self.load_mesh_file(path)
        faces = polydata.faces.reshape(-1, 4)[:, 1:].T
        normal = polydata.point_normals
        pos = polydata.points
        data = Data(
            pos=torch.tensor(pos),
            normal=torch.tensor(normal),
            face=torch.tensor(faces),
            vtkpath = os.path.abspath(path)
        )
        return data

    def lookup_labels(self, subject_id):
        mask = self.metadata_df[self.subject_id_colname] == subject_id
        labels = self.metadata_df.loc[mask]
        return labels

    def len(self):
        return len(self.subject_ids)

    def get(self, idx):
        subject_id = self.subject_ids[idx]
        paths = self.lookup_dict[subject_id]
        labels = self.lookup_labels(subject_id)
        meshes = []
        for substructure in self.substructures:
            mesh = self.get_data_from_polydata(paths[substructure])
            meshes.append(mesh)
        sample = {}
        sample['x'] = meshes
        sample['age'] = torch.from_numpy(np.array(labels['age'], dtype='float32'))
        sample['sex'] = torch.from_numpy(np.array(labels['sex'], dtype='int64'))
        sample['dx'] = torch.from_numpy(np.array(labels['dx'], dtype='int64'))
        sample['y'] = sample[self.target_label]
        return sample

    @property
    def num_features(self) -> int:
        graph = self[0]['x'][0]
        return graph.num_node_features


class MeshDataModule(pl.LightningDataModule):
    def __init__(self, dataset, data_path, metadata_file, target_label, cache_path, train_split, val_split, substructures, train_transform, test_transform, batch_size, num_workers):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = dataset(
            path=data_path,
            substructures=substructures,
            target_label=target_label,
            reload_path=False,
            transform=test_transform,
            metadata_file=metadata_file,
            cache_path=cache_path,
        )

        total_length = len(self.dataset)
        dev_length = int(train_split * total_length)
        test_length = total_length - dev_length
        val_length = int(val_split * dev_length)
        train_length = dev_length - val_length

        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            self.dataset,
            lengths=[train_length, val_length, test_length]
        )

        self.train_set.dataset = copy(self.dataset)
        self.train_set.dataset.transform = MultiGraphTransform(transform=train_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
