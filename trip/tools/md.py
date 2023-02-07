import argparse
from sys import stdout

from openmm import *
from openmm.app import *
from openmm.unit import *

from scipy.optimize import minimize
import torch


from trip.data import AtomicData
from trip.data_loading import GraphConstructor
from trip.model import TrIP


def parse_args():
    parser = argparse.ArgumentParser(description='run md simulations using TrIP')
    parser.add_argument('-i', dest='inFile',type=str, help='Path to the directory with the input pdb file')
    parser.add_argument('-o', dest='outFile',type=str, default='/results/md_output.dcd',
                        help='The name of the output file that will be written, default=/results/md_output.pdb')
    parser.add_argument('-s', dest='stepSize',type=float, default=0.5,
                        help='Step size in femtoseconds, default=0.5')
    parser.add_argument('-t', dest='simTime',type=float, default=1.0,
                        help='Simulation time in picoseconds, default=1')
    parser.add_argument('-m', dest='modelFile',type=str, default='/results/trip_vanilla.pth',
                        help='.pth model file name, default=')
    args = parser.parse_args()
    return args


class TrIPModule(torch.nn.Module):
    def __init__(self, model_file, species):
        super(TrIPModule, self).__init__() 
        self.model = TrIP.load(model_file, map_location='cuda:0')
        self.graph_constructor = GraphConstructor(cutoff=self.model.cutoff)

        symbol_list = AtomicData.get_atomic_symbols_list()
        symbol_dict = {symbol: i+1 for i, symbol in enumerate(symbol_list)}
        self.species_tensor = torch.tensor([symbol_dict[atom] for atom in species], dtype=torch.int, device='cuda')
	
    def forward(self, positions, box_size, forces=True):
        graph = self.graph_constructor.create_graphs(positions, box_size) # Cutoff for 5-12 model is 3.0 A
        graph.ndata['species'] = self.species_tensor

        if forces:
            energy, forces = self.model(graph, forces=forces, create_graph=False)
            return energy.item(), forces
        else:
            energy = self.model(graph, forces=forces, create_graph=False)
            return energy.item()


def energy_function(positions):
    global count
    count+=1
    print(f'Energy Function called: {count}')
    positions = torch.tensor(positions, dtype=torch.float, device='cuda').reshape(-1,3)
    energy = sm(positions, box_size=box_size, forces=False)
    print(f'Energy: {energy:0.5f}')
    return energy

def jacobian(positions):
    print('Force Function called')
    positions = torch.tensor(positions, dtype=torch.float, device='cuda').reshape(-1,3)
    _, forces = sm(positions, box_size=box_size)
    norm_forces = torch.norm(forces)
    print(f'Forces: {norm_forces:0.5f}')
    return -forces.detach().cpu().numpy().flatten()


args = parse_args()

#initialize parameters
in_file = args.inFile
out_file = args.outFile
step_size = args.stepSize
simulation_time = args.simTime
model_file = args.modelFile

pdbf = PDBFile(in_file)
topo = pdbf.topology

system = System()
for atom in topo.atoms():
    system.addParticle(atom.element.mass)

trip_force = CustomExternalForce('-fx*x-fy*y-fz*z')
system.addForce(trip_force)
trip_force.addPerParticleParameter('fx')
trip_force.addPerParticleParameter('fy')
trip_force.addPerParticleParameter('fz')

species = []
for atom in topo.atoms():
    sym = atom.element.symbol
    species.append(sym)

sm = TrIPModule(model_file, species)

count = 0


pos = torch.tensor(pdbf.getPositions(asNumpy=True)/angstrom, dtype=torch.float, device='cuda')


box_size = topo.getUnitCellDimensions() / angstrom
box_size = torch.tensor(box_size, dtype=torch.float, device='cuda')

res = minimize(energy_function, pos.cpu().numpy().flatten(), method='CG', jac=jacobian)
newpos = torch.tensor(res.x, dtype=torch.float, device='cuda').reshape(-1,3)

pdbfile.PDBFile.writeFile(topo, newpos*angstrom, open('minimized', 'w'))

energy, forces = sm(newpos, box_size=box_size)


index = 0
for atom in topo.atoms():
    trip_force.addParticle(index, (forces[index][0].item()*627.5, forces[index][1].item()*627.5, forces[index][2].item()*627.5)*kilocalorie_per_mole/angstrom)
    index+=1

newpos = pos
temperature = 298.0 * kelvin
integrator = LangevinIntegrator(temperature, 1/picosecond, step_size*femtosecond)
simulation = Simulation(topo, system, integrator)


positions = newpos.tolist()*angstrom
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)


state = simulation.context.getState(getPositions=True, getVelocities=True)

#Run simulation and do force calculations
simulation.reporters.append(DCDReporter(out_file, 1))
simulation.reporters.append(StateDataReporter(stdout, 1, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))

num_steps = int((simulation_time/step_size)*1000)

for i in range(num_steps):
    simulation.step(1)
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    newpos = torch.tensor([[pos.x, pos.y, pos.z] for pos in positions],
                          dtype=torch.float, device='cuda')*10.0 # Nanometer to Angstrom
    
    energy, forces = sm(newpos, box_size)
    
    forces *= 627.5
    forces = forces * kilocalorie_per_mole / angstrom
    
    #import pdb; pdb.set_trace()

    for index, atom in enumerate(topo.atoms()):
        trip_force.setParticleParameters(index, index, forces[index])
        
    trip_force.updateParametersInContext(simulation.context)
