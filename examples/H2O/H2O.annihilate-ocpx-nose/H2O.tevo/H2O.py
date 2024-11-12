from os.path import abspath

prefix = 'H2O'
memory = 250.0E9
dump_inputs = True
T0_PATH = '..'
prev_logbook = abspath(T0_PATH + '/H2O.lb')
complex_MPS_type = 'full'

atoms = 'logbook'
basis = 'logbook'
group = 'logbook'
wfn_sym = 'B1'
orb_path = 'logbook'
orb_order = 'logbook:orb_order_id'

nCore = 'logbook'
nCAS = 'logbook'
nelCAS = 7       # The initial state has one less electron due to annihilation operator.
twos = 1

do_groundstate = False
do_annihilate = False

do_timeevo = True
if do_timeevo:
    te_inmps_dir = abspath(T0_PATH + '/H2O.ann-mps')
    te_max_D = 600
    te_inmps_cpx = True
    tinit = 0.0
    tmax = 20.0
    fdt = 0.04
    dt = [fdt/4]*4 + [fdt/2]*2 + [fdt]
    te_method = 'tdvp'
    krylov_size = 5
    te_sample = ('delta', 5*dt[-1])
