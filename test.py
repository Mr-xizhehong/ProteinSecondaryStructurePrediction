import numpy as np
from Bio.PDB import PDBParser

def parse_pdb(file_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('protein', file_path)
    return structure

def extract_atom_coordinates(structure):
    atom_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coords.append(atom.get_coord())
    return atom_coords

pdb_file_path = r"C:\Users\java~python\Desktop\P42212_archive-PDB\pdb1b9c.ent"
structure = parse_pdb(pdb_file_path)
atom_coordinates = extract_atom_coordinates(structure)

# 打印原子坐标
atom_coordinates = np.array(atom_coordinates)
print(atom_coordinates.shape)
