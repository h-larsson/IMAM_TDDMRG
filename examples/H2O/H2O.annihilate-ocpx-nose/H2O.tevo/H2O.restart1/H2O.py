from os.path import abspath

prefix = 'H2O'
memory = 210.0E9
dump_inputs = True
T0_PATH = '..'
prev_logbook = abspath(T0_PATH + '/H2O.lb')
complex_MPS_type = 'logbook'

atoms = 'logbook'
basis = 'logbook'
group = 'logbook'
wfn_sym = 'logbook'
orb_path = 'logbook'
orb_order = 'logbook:orb_order_id'

nCore = 'logbook'
nCAS = 'logbook'
nelCAS = 'logbook'
twos = 'logbook'

do_groundstate = False
do_annihilate = False

do_timeevo = True
if do_timeevo:
    te_inmps_dir = abspath(T0_PATH + '/H2O.mps_t')
    te_inmps_fname = 'mps_info.bin'
    te_max_D = 'logbook'
    te_inmps_cpx = 'logbook'
    tinit = 17.8
    tmax = 'logbook'
    fdt = 0.04
    dt = [fdt]
    te_method = 'logbook'
    krylov_size = 'logbook'
    te_sample = 'logbook'

    mps_act0_dir = 'logbook'
    mps_act0_fname = 'logbook'
    mps_act0_cpx = 'logbook'
    mps_act0_multi = 'logbook'
