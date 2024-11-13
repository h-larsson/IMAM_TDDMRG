import numpy as np
from pyscf import symm
from TDDMRG_CM.utils import util_logbook, util_atoms
from TDDMRG_CM.orbs_generate import hdm_orb, analyze_orbs

#================== BEGIN INPUTS ==================#
'''
In this script, we will calculate the DM-adapted and hole-DM-adapted
orbitals. For example, let's assume that the desired number of active orbitals
is 18. nbase is chosen to be equal to the number of active orbitals with
occupancies close to 2 or 1, for water, it is 4. If the desired nCAS is still
well below the total number of orbitals linear dependency usually will not
appear, in which case nav = nCAS - nbase = 18 - 4 = 4.
'''

nbase = 4                                          # The number of base orbitals.
nav = 14                                           # The number of correction orbitals.
rdm0_path = '../../../../H2O.gs-mps/GS_1pdm.npy'   # Path to the spin-1RDM of the ground state of neutral H2O.
te_logbook_path = '../../H2O.lb'                   # Path to logbook file of the TDDMRG simulation.
imtype = 'ignore'                                  # Ignore the imaginary part of time-dependent 1RDMs.
save_path = '.'
nelCAS = 7                                         # Number of active electrons in the cation regardless of singlet-embedding is used or not.
tdir = ('../../H2O.sample',
        '../../H2O.restart1/H2O.sample')
# tdir contains the <prefix>.sample directories of the different parts of TDDMRG
# simulations. More than one part can exist if there are restarts.
#================== END INPUTS ==================#



#==== General setup ====#
logbook = util_logbook.read(te_logbook_path)
mol = util_atoms.mole(logbook)
prefix = logbook['prefix']
orb_path = logbook['orb_path']
nCore = logbook['nCore']
nCAS = logbook['nCAS']


#==== Step 1: Calculate base orbitals ====#
print('\n\n')
print('====================')
print('>>>>>> Step 1 <<<<<<')
print('====================\n')
orb = np.load(logbook['orb_path'])
orbc = orb[:,0:nCore].copy()                       # core orbitals
orba = orb[:,nCore:nCore+nCAS].copy()              # active orbitals
syma = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orba)    # irreps of active orbitals
srdm0 = np.load(rdm0_path)                         # spin-1RDM of the ground state of neutral molecule
_, orbn_ = symm.eigh(np.sum(srdm0, axis=0), syma)                    # orbn_ = natural orbitals in active orbitals basis.
orbn_ = orbn_[:,::-1]
orbb = (orba @ orbn_)[:,nCore:nCore+nbase]         # orbb = base orbitals


#==== Step 2: Calculate averaged natural charge orbitals ====#
print('\n\n')
print('====================')
print('>>>>>> Step 2 <<<<<<')
print('====================\n')
orbav, _, _ = hdm_orb.average(srdm0, imtype=imtype, nelCAS=nelCAS,
                              tdir=tdir, logbook=logbook)
orbdm = np.hstack((orbc, orbav))
dm_name = save_path + '/' + prefix + '.dm'
np.save(dm_name, orbdm)
print('\nvvvvvvvvvv')
print('Density-matrix-adpated orbitals have been saved to ' + dm_name + '.npy.')
print('^^^^^^^^^^\n')

#==== Analyses ====#
print('\n\nDM-adapted orbitals AO coeffients:')
analyze_orbs.analyze(mol, orbdm, None, None)
print('\n\nDM-adapted orbital multipole components:')
analyze_orbs.analyze_multipole(mol, orbdm)
print('\n\nDM-adapted orbital atomic populations:')
analyze_orbs.analyze_population(mol, orbdm, 'low')


#==== Step 3: Project averaged natural charge orbitals ====#
#====    onto the orthogonal space of base orbitals    ====#
print('\n\n')
print('====================')
print('>>>>>> Step 3 <<<<<<')
print('====================\n')
orbcr, orbv = hdm_orb.correction(orbb, orbav, nav, mol=mol, nCore=nCore,
                                 corb=orbc)        # orbv = virtual/excitation orbitals
orbhdm = np.hstack((orbc, orbb, orbcr, orbv))
hdm_name = save_path + '/' + prefix + '.hdm'
np.save(hdm_name, orbhdm)
print('\nvvvvvvvvvv')
print('Hole-density-matrix-adpated orbitals have been saved to ' + hdm_name + '.npy.')
print('^^^^^^^^^^\n')

#==== Analyses ====#
print('\n\nHole-DM-adapted orbitals AO coeffients:')
analyze_orbs.analyze(mol, orbhdm, None, None)
print('\n\nHole-DM-adapted orbital multipole components:')
analyze_orbs.analyze_multipole(mol, orbhdm)
print('\n\nHole-DM-adapted orbital atomic populations:')
analyze_orbs.analyze_population(mol, orbhdm, 'low')

