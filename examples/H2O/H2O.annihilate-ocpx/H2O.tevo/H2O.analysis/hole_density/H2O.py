import numpy as np
from TDDMRG_CM.observables import hole_dens
from TDDMRG_CM.utils import util_logbook


#================== BEGIN INPUTS ==================#
logbook_path = '../../H2O.lb'
logbook = util_logbook.read(logbook_path)
nelCAS = 7

rdm0_path = '../../../../H2O.gs-mps/GS_1pdm.npy'
tevo_dir = ('../../H2O.sample',
            '../../H2O.restart1/H2O_c.sample')

nc = (40, 40, 40)
#================== END INPUTS ==================#




rdm0 = np.load(rdm0_path)

print('\n\n\n')
print('*** Printing hole density in 3D space ***')
hole_dens.eval_volume(rdm0, nc, tdir=tevo_dir, nelCAS=nelCAS, tnorm0=False, tnorm1=True,
                      prefix='hdens_vol', logbook=logbook)

