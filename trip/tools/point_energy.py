import argparse
import torch
from trip.tools import TrIPModule
import MDAnalysis as mda
from trip.tools.utils import get_species

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calculate point energies from a .pdb file')
parser.add_argument('--pdb', help='Name of the input .pdb file (without extension)')
parser.add_argument('--model_file', type=str, default='/results/model_ani2x_v1_1.pth', help='Path to model file, default=/results/model_ani2x_v1_1.pth')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use, default=0')
args = parser.parse_args()

device = f'cuda:{args.gpu}'
torch.cuda.set_device(device)

universe = mda.Universe(args.pdb)
universe.atoms.guess_bonds(vdwradii={'Cl': 1.735})
symbols = universe.atoms.elements
species = get_species(symbols)

boxsize = torch.full((3,), float('inf'), dtype=torch.float, device=device)
pos = torch.tensor(universe.atoms.positions, dtype=torch.float, device=device)

trip_module = TrIPModule(species, args.model_file, args.gpu)

pos = trip_module.minimize(pos, boxsize)
energy = trip_module.forward(pos, boxsize, forces=False)

print(energy)
