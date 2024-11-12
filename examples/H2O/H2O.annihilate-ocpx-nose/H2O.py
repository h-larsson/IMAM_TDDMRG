import numpy as np
from os.path import abspath

prefix = 'H2O'
GS_PATH = '..'        # Relative path of the ground state calculation.
prev_logbook = abspath(GS_PATH + '/H2O.lb')    # Convert to an absolute path right here---another great thing of having an input file as a Pythons script.
complex_MPS_type = 'logbook'

atoms = 'logbook'
basis = 'logbook'
group = 'logbook'
wfn_sym = 'logbook'
orb_path = 'logbook'    # No need for abspath here because orb_path from the previous logbook contains the full path already.
orb_order = 'logbook:orb_order_id'

nCore = 'logbook'
nCAS = 23
nelCAS = 'logbook'
twos = 'logbook'

do_groundstate = False

do_annihilate = True
if do_annihilate:
    ann_sp = True
    ann_orb = np.zeros(nCAS)
    ann_orb[3] = ann_orb[9] = 1/np.sqrt(2)
    D_ann_fit = [100]*4 + [300]*4 + [500]*4 + [800]*4 + [600]*4 + [400]
    ann_inmps_dir = abspath(GS_PATH + '/H2O.gs-mps')
    ann_outmps_dir = abspath('./' + prefix + '.ann-mps')
    ann_out_cpx = True

do_timeevo = False
