import logging

import numpy as np
import psi4
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdDistGeom import ETKDGv3
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

ATOM_DECODER = ["H", "C", "N", "O", "F"]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

BOND_DICT = [
    None,
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]


def build_molecule(atom_types, edge_types, atom_decoder):
    atom_types_list = atom_types.tolist()
    mol = Chem.RWMol()

    for at_idx in atom_types_list:
        mol.AddAtom(Chem.Atom(atom_decoder[int(at_idx)]))

    e = torch.triu(edge_types).cpu().numpy()
    e[e >= 5] = 0

    rows, cols = np.nonzero(e)
    for i, j in zip(rows, cols):
        if i != j:
            mol.AddBond(int(i), int(j), BOND_DICT[int(e[i, j])])

    return mol


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        import re as _re

        msg = str(e)
        part = msg[msg.find("#") :]
        atomid_valence = list(map(int, _re.findall(r"\d+", part)))
        return False, atomid_valence


def build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder):
    mol = Chem.RWMol()

    for a in atom_types.tolist():
        mol.AddAtom(Chem.Atom(atom_decoder[int(a)]))

    edge_types = torch.triu(edge_types)
    edge_types = torch.where(edge_types >= 5, torch.zeros_like(edge_types), edge_types)
    bonds = torch.nonzero(edge_types, as_tuple=False)

    for i, j in bonds.tolist():
        if i != j:
            mol.AddBond(int(i), int(j), BOND_DICT[int(edge_types[i, j])])

    ok, info = check_valency(mol)
    if ok:
        return mol

    idx, val = info
    atom = mol.GetAtomWithIdx(idx)
    atomic_num = atom.GetAtomicNum()

    if atomic_num in (7, 8, 16) and (val - ATOM_VALENCY[atomic_num] == 1):
        atom.SetFormalCharge(1)

    return mol


class Evaluator:
    def __init__(self, nthreads=4, mem_gb=5, level="b3lyp/6-31G*"):
        self.level = level

        # Disable logging
        RDLogger.DisableLog("rdApp.*")

        psi4.core.set_output_file("/dev/null", False)
        psi4.set_options({"PRINT": 0})

        for name in [
            "psi4",
            "psi4.driver",
            "psi4.driver.task_planner",
            "psi4.driver.driver",
        ]:
            logging.getLogger(name).setLevel(logging.CRITICAL)

        psi4.set_num_threads(nthreads)
        psi4.set_memory(f"{mem_gb}GB")

    def compute_validity(self, generated):
        valid_list = []
        num_components = []

        for atom_types, edge_types in generated:
            mol = build_molecule(atom_types, edge_types, ATOM_DECODER)

            try:
                frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                valid_list.append(True)
                num_components.append(len(frags))
            except Exception:
                valid_list.append(False)
                num_components.append(0)

        return sum(valid_list) / len(valid_list)

    def cond_sample_metric(self, samples, input_properties, num_eval=5):
        samples = samples[:num_eval]
        input_properties = input_properties[:num_eval]

        true_props = []
        energies = []

        for i, (atoms, edges) in enumerate(samples):
            mol = build_molecule_with_partial_charges(atoms, edges, ATOM_DECODER)

            # Sanitize fails → invalid molecule
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue

            # 3D + MMFF
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1

            try:
                EmbedMolecule(mol, params)
                MMFFOptimizeMolecule(mol)
            except Exception:
                continue

            try:
                conf = mol.GetConformer()
                # Charge and multiplicity
                charge = Chem.GetFormalCharge(mol)
                radical_e = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
                spin_mult = int(2 * (radical_e / 2) + 1)
            except Exception:
                continue

            # Psi4 geometry
            geom = f"{charge} {spin_mult}"
            for idx, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(idx)
                geom += f"\n{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}"

            try:
                e = psi4.energy(self.level, molecule=psi4.geometry(geom))
            except Exception:
                continue

            energies.append(e)
            true_props.append(input_properties[i].reshape(1, -1))

        if len(true_props) == 0:
            return None

        true_props = torch.cat(true_props, dim=0)
        pred = torch.FloatTensor(energies).unsqueeze(1)

        mae = torch.mean(torch.abs(pred - true_props))
        return mae
