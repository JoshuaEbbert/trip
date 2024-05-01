import logging

from scipy.optimize import minimize
import torch
from torch.autograd.functional import hessian
import numpy as np

import torchani

from trip.data_loading import GraphConstructor
from trip.model import TrIP


class TrIPModule(torch.nn.Module): 
    def __init__(self, species, model_file, gpu, constraints=[], **vars):
        super().__init__()
        self.device = f'cuda:{gpu}'
        self.species_tensor = torch.tensor(species, dtype=torch.long, device=self.device)
        self.constraints = constraints
        if model_file == 'ani2x':  
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
            self.forward = self.ani_forward
        else:
            self.model = TrIP.load(model_file, map_location=self.device)
            self.graph_constructor = GraphConstructor(cutoff=self.model.cutoff)
            self.forward = self.trip_forward

    def trip_forward(self, pos, boxsize, forces=True):
        graph = self.graph_constructor.create_graphs(pos, boxsize)  # Cutoff for 5-12 model is 3.0 A
        graph.ndata['species'] = self.species_tensor

        if forces:
            energy, forces = self.model(graph, forces=True)
            return energy.item(), forces
        else:
            energy = self.model(graph, forces=False)
            return energy.item()

    def ani_forward(self, pos, boxsize, forces=True):
        # Only supports boxsize == inf
        pos.requires_grad_(True)
        energy = self.model((self.species_tensor.unsqueeze(0), pos.unsqueeze(0))).energies.sum()
        if forces:
            forces = -torch.autograd.grad(energy.sum(), pos)[0]
            return energy.item(), forces
        else:
            return energy.item()

    def energy_np(self, pos, boxsize):
        pos = pos.clone().detach().to(self.device).reshape(-1,3)
        with torch.no_grad():
            energy = self.forward(pos, boxsize=boxsize, forces=False)
        energy += self.calc_constraints(pos)
        if self.forward == self.ani_forward:
            try:    
                energy = energy.item()
            except AttributeError: 
                pass

        return energy

    def jac_np(self, pos, boxsize):
        pos = torch.tensor(pos, dtype=torch.float, device=self.device).reshape(-1,3)
        _, forces = self.forward(pos, boxsize=boxsize)
        pos.requires_grad_(True)   
        error = self.calc_constraints(pos)
        if error == 0:
            jac = torch.zeros_like(pos)
        else:
            jac = torch.autograd.grad(error, pos, create_graph=True)[0]
        jac -= forces
        pos.requires_grad_(False)  
        return jac.detach().cpu().numpy().flatten()

    def hess(self, pos, boxsize):  
        def energy(pos):
            graph = self.graph_constructor.create_graphs(pos.reshape(-1,3), boxsize)
            graph.ndata['species'] = self.species_tensor
            return self.model(graph, forces=False)
        return hessian(energy, pos.flatten())

    def minimize(self, pos, boxsize, method='CG'):
        def energy_np_cpu(pos_flat, boxsize):
            pos_flat = torch.from_numpy(pos_flat.astype(np.float32)).to(self.device)
            energy = self.energy_np(pos_flat, boxsize)
            return energy.cpu().numpy() if isinstance(energy, torch.Tensor) else energy

        sol = minimize(energy_np_cpu, pos.cpu().numpy().flatten(), args=boxsize, method=method, jac=self.jac_np)
        return torch.tensor(sol.x, dtype=torch.float, device=self.device).reshape(-1,3)

    def calc_constraints(self, pos):
        return sum([constraint(pos) for constraint in self.constraints])

