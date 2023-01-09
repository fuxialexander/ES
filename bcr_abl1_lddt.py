#%%
from Bio.PDB import PDBParser
bioparser = PDBParser()
# %%
structure = bioparser.get_structure('dimer', 'dimer_structures/' + 'BCR_ABL1' + '.pdb')
# %%
lddt = []
for r in structure.get_residues():
    lddt.append([a.get_bfactor() for a in r.get_atoms()][0])
# %%
import numpy as np
# %%
np.save("dimer_structures/BCR_ABL1.lddt.npy", np.array(lddt))
# %%
