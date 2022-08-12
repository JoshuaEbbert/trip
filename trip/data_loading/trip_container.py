

import pathlib
from typing import List, Literal, Optional
import h5py

import torch
from torch import Tensor


class TrIPContainer:
    def __init__(self, file_name: Optional[pathlib.Path] = None):
        super().__init__()
        self.train_species_data, self.val_species_data, \
            self.test_species_data = [], [], []
        self.train_pos_data, self.val_pos_data, \
            self.test_pos_data = [], [], []
        self.train_energy_data, self.val_energy_data, \
            self.test_energy_data = [], [], []
        self.train_forces_data, self.val_forces_data,\
             self.test_forces_data = [], [], []
        self.train_box_size_data, self.val_box_size_data,\
             self.test_box_size_data = [], [], []

        if file_name is not None:
            self.load_data(file_name)

    def load_data(self, file_name: pathlib.Path):
        with h5py.File(file_name, 'r') as f:
            for group_name in ['train', 'val', 'test']:
                self.set_data(group_name, *self._read_group(f, group_name))

    def _read_group(self, f: h5py.File,
                    group_name: Literal['train', 'val', 'test']):
        group = f[group_name]
        species_data = []
        pos_data = []
        energy_data = []
        forces_data = []
        box_size_data = []
        for molecule in group.values():
            species_data.append(torch.tensor(molecule['species'][:], dtype=torch.uint8))
            pos_data.append(torch.tensor(molecule['pos'][:], dtype=torch.float32))
            energy_data.append(torch.tensor(molecule['energy'][:], dtype=torch.float32))
            forces_data.append(torch.tensor(molecule['forces'][:], dtype=torch.float32))
            box_size_data.append(torch.tensor(molecule['box_size'][:], dtype=torch.float32))
        return species_data, pos_data, energy_data, forces_data, box_size_data

    def set_data(self, group_name: Literal['train', 'val', 'test'],
                 species_data: List[Tensor],
                 pos_data: List[Tensor],
                 energy_data: List[Tensor],
                 forces_data: List[Tensor],
                 box_size_data: List[Tensor],
                 ):
        data = species_data, pos_data, energy_data, forces_data, box_size_data
        if group_name == 'train':
            self.train_species_data, self.train_pos_data, \
                self.train_energy_data, self.train_forces_data, \
                self.train_box_size_data = data
        elif group_name == 'val':
            self.val_species_data, self.val_pos_data, \
                self.val_energy_data, self.val_forces_data, \
                self.val_box_size_data = data
        elif group_name == 'test':
            self.test_species_data, self.test_pos_data, \
                self.test_energy_data, self.test_forces_data, \
                self.test_box_size_data = data

    def get_data(self, name: Literal['train', 'val', 'test']):
        if name == 'train':
            return self.train_species_data, self.train_pos_data, \
                self.train_energy_data, self.train_forces_data, \
                self.train_box_size_data
        elif name == 'val':
            return self.val_species_data, self.val_pos_data, \
                self.val_energy_data, self.val_forces_data, \
                self.val_box_size_data
        elif name == 'test':
            return self.test_species_data, self.test_pos_data, \
                self.test_energy_data, self.test_forces_data, \
                self.test_box_size_data

    def save_data(self, file_name: pathlib.Path):
        with h5py.File(file_name, 'w') as f:
            for group_name in ['train', 'val', 'test']:
                self._add_group(f, group_name, *self.get_data(group_name))

    def _add_group(self, f: h5py.File, 
                   group_name: Literal['train', 'val', 'test'], 
                   species_data: List[Tensor],
                   pos_data: List[Tensor],
                   energy_data: List[Tensor],
                   forces_data: List[Tensor],
                   box_size_data: List[Tensor],
                   ):
        group = f.create_group(group_name)
        for i in range(len(species_data)):
            molecule = group.create_group(str(i))
            molecule.create_dataset('species', data=species_data[i])
            molecule.create_dataset('pos', data=pos_data[i])
            molecule.create_dataset('energy', data=energy_data[i])
            molecule.create_dataset('forces', data=forces_data[i])
            molecule.create_dataset('box_size', data=box_size_data[i])

    def get_species_list(self):
        species = set()
        for species_list in self.train_species_data:
            species.union(set(species_list.tolist()))
        return list(species)
