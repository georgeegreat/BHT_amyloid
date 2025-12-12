from collections import defaultdict
from typing import List, Dict
import pickle
import xml.etree.ElementTree as ET
import requests
from openai import OpenAI
from structure import Structure
from rdkit import Chem
import selfies as sf
import numpy as np
from rdkit.Chem import Descriptors
import pubchempy as pcp
from utils.sdf_parser import SDFParser
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import RDKFingerprint

# SMILES, ex. : C1=CC=CC=C1
# SELFIES, ex. : [C][=C][C][=C][C][=C][Ring1][=Branch1]
# SMARTS, ex. : [C:1]=[O,N:2]>>*[C:1][*:2]

# убрать формат, держать все в smiles с возможностью вызова других типов


class Molecule(Structure):
    def __init__(self, name: str, sequence: str):  # smiles
        super().__init__(name, sequence)
        self.molecule = Chem.MolFromSmiles(self.sequence)

        self.atom_dict = defaultdict(lambda: {})
        self.bond_dict = defaultdict(lambda: {})
        self.fingerprint_dict = defaultdict(lambda: {})
        self.edge_dict = defaultdict(lambda: {})

        try:
            self.cid = pcp.get_compounds(self.sequence, 'smiles')[0].cid # try with name
        except:
            self.cid = pcp.get_compounds(self.name, 'name')[0].cid



    def get_name(self):
        cmp = pcp.get_compounds(Chem.MolToSmiles(self.molecule), 'smiles')
        print(cmp[0].iupac_name)
        return cmp[0].iupac_name

    def get_desc(self):
        if self.name != "":
            req_txt = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{self.name}/description/XML"
        else:
            req_txt = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.cid}/description/XML"
        req = requests.get(req_txt)
        print(req.text, req_txt)
        try:
            self.pbchm_desc = ET.fromstring(req.text)[1][1].text
        except:
            self.pbchm_desc = ""
        return self.pbchm_desc

    def get_main_from_desc(self, key, prompt_inp=None, sys_prompt_inp=None):  # this is only for deepseek usage
        if self.pbchm_desc == "":
            return "Нет описания"
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

        if prompt_inp is None:
            prompt = f"""
                    Выдели из этого описания самого важное {self.pbchm_desc}
                    """
        else:
            prompt = prompt_inp

        if sys_prompt_inp is None:
            SYSTEM_PROMPT = """
                    Ты работаешь лабораторным ассистентом и тебе надо помочь собрать качественный датасет на основе описания химических веществ 
                    """
        else:
            SYSTEM_PROMPT = sys_prompt_inp

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content


    def fingerprints(self):
        return RDKFingerprint(self.sequence)

    def similatity(self, other, typ='tanimoto'):
        if type(other) != Molecule:
            raise ValueError
        if typ == "tanimoto":
            return DataStructs.TanimotoSimilarity(RDKFingerprint(self.sequence), other.fingerprints())

    def get_3d_coords(self):  # https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/SDF?record_type=3d
        req = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{self.cid}/SDF?record_type=3d"
        sdf_file = requests.get(req).text
        return SDFParser().process_data(sdf_file)

    def create_adjacency(self):
        adjacency = Chem.GetAdjacencyMatrix(self.molecule)
        return np.array(adjacency)

    def create_atoms(self):
        atoms = [a.GetSymbol() for a in self.molecule.GetAtoms()]
        for a in self.molecule.GetAromaticAtoms():
            i = a.GetIdx()
            atoms[i] = (atoms[i], 'aromatic')
        atoms = [self.atom_dict[a] for a in atoms]
        return np.array(atoms)

    def create_ij_bond_dict(self):
        i_jbond_dict = defaultdict(lambda: [])
        for b in self.molecule.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bond = self.bond_dict[str(b.GetBondType())]
            i_jbond_dict[i].append((j, bond))
            i_jbond_dict[j].append((i, bond))
        return i_jbond_dict

    @staticmethod
    def get_smiles_from_name(name):
        req_txt = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        return requests.get(req_txt).text

    @staticmethod
    def smiles_to_selfies(smiles):
        return sf.encoder(smiles)


    def get_selfies(self):
        return sf.encoder(self.sequence)

    def extract_fingerprints(self, atoms, i_jbond_dict, radius):
        if (len(atoms) == 1) or (radius == 0):
            fingerprints = [self.fingerprint_dict[a] for a in atoms]
        else:
            nodes = atoms
            i_jedge_dict = i_jbond_dict

            for _ in range(radius):

                fingerprints = []
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    fingerprints.append(self.fingerprint_dict[fingerprint])
                nodes = fingerprints
                _i_jedge_dict = defaultdict(lambda: [])
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = self.edge_dict[(both_side, edge)]
                        _i_jedge_dict[i].append((j, edge))
                i_jedge_dict = _i_jedge_dict

        return np.array(fingerprints)

    def smiles_to_descriptors(self):
        mol = Chem.MolFromSmiles(self.sequence)
        descriptors = []
        descriptor_names = ['exactmw', 'amw', 'lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumHeavyAtoms', 'NumAtoms', 'NumHeteroatoms', 'NumAmideBonds', 'FractionCSP3', 'NumRings', 'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings', 'NumHeterocycles', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'NumSpiroAtoms', 'NumBridgeheadAtoms', 'NumAtomStereoCenters', 'NumUnspecifiedAtomStereoCenters', 'labuteASA', 'tpsa', 'CrippenClogP', 'CrippenMR', 'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v', 'chi0n', 'chi1n', 'chi2n', 'chi3n', 'chi4n', 'hallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'Phi']
        # list(rdMolDescriptors.Properties.GetAvailableProperties())
        print("descs=", descriptor_names)
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        if mol:
            descriptors = np.array(get_descriptors.ComputeProperties(mol))
        return {descriptor_names[i]: descriptors[i] for i in range(len(descriptors))}

    @staticmethod
    def smiles_to_descriptors_static(mol):
        mol = Chem.MolFromSmiles(mol)
        descriptors = []
        descriptor_names = ['exactmw', 'amw', 'lipinskiHBA', 'lipinskiHBD', 'NumRotatableBonds', 'NumHBD', 'NumHBA',
                            'NumHeavyAtoms', 'NumAtoms', 'NumHeteroatoms', 'NumAmideBonds', 'FractionCSP3', 'NumRings',
                            'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings', 'NumHeterocycles',
                            'NumAromaticHeterocycles', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles',
                            'NumSpiroAtoms', 'NumBridgeheadAtoms', 'NumAtomStereoCenters',
                            'NumUnspecifiedAtomStereoCenters', 'labuteASA', 'tpsa', 'CrippenClogP', 'CrippenMR',
                            'chi0v', 'chi1v', 'chi2v', 'chi3v', 'chi4v', 'chi0n', 'chi1n', 'chi2n', 'chi3n', 'chi4n',
                            'hallKierAlpha', 'kappa1', 'kappa2', 'kappa3', 'Phi']
        # list(rdMolDescriptors.Properties.GetAvailableProperties())
        print("descs=", descriptor_names)
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        if mol:
            descriptors = np.array(get_descriptors.ComputeProperties(mol))
        return {descriptor_names[i]: descriptors[i] for i in range(len(descriptors))}


if __name__ == "__main__":
    m = Chem.MolFromSmiles("Nc1nc(N)[nH+]c(NCCCCNc2c3ccccc3[nH+]c3c(C(=O)NCCC[NH2+]CCC[NH2+]CCCNC(=O)c4cccc5c(NCCCCNc6nc(N)[nH+]c(N)n6)c6ccccc6[nH+]c45)cccc23)n1")
    print(Chem.MolToSmiles(m))
