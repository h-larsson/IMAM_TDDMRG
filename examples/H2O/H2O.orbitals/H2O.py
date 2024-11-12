import numpy as np


#== General ==#
prefix = 'H2O'

#== System ==#
inp_coordinates = \
            '''
            O   0.000   0.000   0.107;
            H   0.000   0.785  -0.427;
            H   0.000  -0.785  -0.427;
            '''
inp_basis = 'cc-pvdz'
inp_symmetry = 'C2v'
charge = 0
twosz = 0
wfnsym = 0
source = 'rhf'

localize = False
