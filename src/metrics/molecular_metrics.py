import numpy as np
import torch
from rdkit import Chem

ATOM_DECODER = ["H", "C", "N", "O", "F"]
BOND_DICT = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def build_molecule(atom_types, edge_types, atom_decoder):
    # Convert tensors → Python lists once (avoids thousands of .item() calls)
    atom_types_list = atom_types.tolist()

    # Pre-allocate molecule
    mol = Chem.RWMol()

    # Add atoms
    for at_idx in atom_types_list:
        mol.AddAtom(Chem.Atom(atom_decoder[at_idx]))

    # Upper triangular part (avoid double-adding bonds)
    e = torch.triu(edge_types).cpu().numpy()

    # Mask out virtual bonds (>=5)
    e[e >= 5] = 0

    # Extract real bonds only
    rows, cols = np.nonzero(e)
    for i, j in zip(rows, cols):
        if i == j:
            continue
        mol.AddBond(int(i), int(j), BOND_DICT[e[i, j]])

    return mol


def compute_validity(generated):
    valid_list = []
    num_components = []

    for atom_types, edge_types in generated:
        mol = build_molecule(atom_types, edge_types, ATOM_DECODER)

        try:
            # This performs both fragmentation + sanitization
            frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            valid_list.append(True)
            num_components.append(len(frags))
        except Exception:
            valid_list.append(False)
            num_components.append(0)

    validity_ratio = sum(valid_list) / len(valid_list)
    return validity_ratio
