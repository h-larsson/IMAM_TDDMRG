import numpy as np
from TDDMRG_CM.observables import hole_dens
from TDDMRG_CM.utils import util_logbook


#================== BEGIN INPUTS ==================#
logbook_path = '../../H2O.lb'
logbook = util_logbook.read(logbook_path)
nelCAS = 7          # Number of active electrons in the cation regardless of singlet-embedding is used or not.
rdm0_path = '../../../../H2O.gs-mps/GS_1pdm.npy'
tdir = ('../../H2O.sample',
        '../../H2O.restart1/H2O.sample')
# tdir contains the <prefix>.sample directories of the different parts of TDDMRG
# simulations. More than one part can exist if there are restarts.
nc = (40, 40, 40)
#================== END INPUTS ==================#


srdm0 = np.load(rdm0_path)
print('\n\n\n')
print('*** Printing hole density in 3D space ***')
hole_dens.eval_volume(srdm0, nc, tdir=tdir, nelCAS=nelCAS, tnorm0=False, tnorm1=True,
                      prefix='hdens_vol', logbook=logbook)

