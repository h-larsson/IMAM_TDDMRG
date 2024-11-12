from os.path import abspath

complex_MPS_type = 'hybrid'
prefix = 'H2O'
atoms = \
            '''
            O   0.000   0.000   0.107;
            H   0.000   0.785  -0.427;
            H   0.000  -0.785  -0.427;
            '''
basis = 'cc-pvdz'
group = 'C2v'
wfn_sym = 'A1'
orb_path = abspath('./H2O.orbitals/H2O.orb.npy')

nCore = 1
nCAS = 23
nelCAS = 8
twos = 0

do_groundstate = True
if do_groundstate:
    D_gs = [100]*4 + [250]*4 + [400]
    gs_outmps_dir = './' + prefix + '.gs-mps'
    save_gs_1pdm = True

do_annihilate = False
do_timeevo = False
