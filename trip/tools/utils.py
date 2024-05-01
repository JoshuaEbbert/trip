from trip.data import AtomicData
import os
from openmm.app import PDBFile
from openmm.unit import angstrom

def get_species(symbols):
    symbol_list = AtomicData.get_symbols_list()
    symbol_dict = {symbol: i+1 for i, symbol in enumerate(symbol_list)}
    species = [symbol_dict[symbol] for symbol in symbols]
    return species

def save_pdb(pos, topo, name, out, **args):
    with open(os.path.join(out, name+'.pdb'), 'w') as f:
        PDBFile.writeFile(topo, pos*angstrom, f)
