import numpy as np
from pyscf import symm
from TDDMRG_CM.utils import util_logbook, util_atoms
from TDDMRG_CM.observables import td_hocc


#================== BEGIN INPUTS ==================#
'''
In this script, we will calculate the time-dependent occupancies of the natural
orbitals of the ground state neutral H2O.
'''

prefix = 'H2O'
logbook_path = '../../H2O.lb'
nelCAS = 7                       # Number of active electrons in the cation regardless of singlet-embedding is used or not.
tdir = ('../../H2O.sample',
        '../../H2O.restart1/H2O.sample')
rdm0_path = '../../../../H2O.gs-mps/GS_1pdm.npy'   # Path to the 1RDM of neutral H2O ground state.
#================== END INPUTS ==================#



logbook = util_logbook.read(logbook_path)          # Load the logbook from a TDDMRG simulation.
mol = util_atoms.mole(logbook)                     # Construct the Mole object used in that TDDMRG simulation.
nCore = logbook['nCore']
nCAS = logbook['nCAS']
orba = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]             # Active orbitals.
sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orba)    # Irrep of active orbitals.
srdm0 = np.load(rdm0_path)                         # spin-1RDM of neutral H2O ground state in active orbitals basis.
occn, orbn_ = symm.eigh(np.sum(srdm0, axis=0), sym)                 # orbn_ is the nat. orbitals in active orbitals basis
orbn = orba @ orbn_                                # Transform from active orbitals basis to AO basis.
td_hocc.calc(orbn, srdm0, tdir=tdir, nelCAS=nelCAS, prefix=prefix, logbook=logbook)

print('\nvvvvvvvvvv')
print('Time-dependent occupancies have been printed into ' + prefix + td_hocc.EXT1 + '.')
print('^^^^^^^^^^\n')
