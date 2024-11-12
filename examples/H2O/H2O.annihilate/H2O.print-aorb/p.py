
import numpy as np
from pyscf.tools import cubegen
from IMAM_TDDMRG.utils import util_logbook, util_atoms


lb = util_logbook.read('../../H2O.lb')
mol = util_atoms.mole(lb)
mo = np.load('../../H2O.orbitals/H2O.orb.npy')

c = 1/np.sqrt(2.0)
v = c*mo[:,4] + c*mo[:,10]

cubegen.orbital(mol, 'aorb.cube', v)


