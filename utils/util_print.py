import numpy as np
try:
    from block2.su2 import MPICommunicator
    hasMPI = True
except ImportError:
    hasMPI = False

if hasMPI:
    MPI = MPICommunicator()
else:
    class _MPI:
        rank = 0
    MPI = _MPI()
    


#################################################
def printDummyFunction(*args, **kwargs):
    """ Does nothing"""
    pass
#################################################


#################################################
def getVerbosePrinter(verbose,indent="",flush=False):
    if verbose:
        if flush:
            def _print(*args, **kwargs):
                kwargs["flush"] = True
                print(indent,*args,**kwargs)
        else:
            def _print(*args, **kwargs):
                print(indent, *args, **kwargs)
    else:
        _print = printDummyFunction
    return _print        
#################################################


#################################################
if hasMPI:
    r0 = (MPI.rank==0)
else:
    r0 = True

_print = getVerbosePrinter(r0, flush=True)
print_i2 = getVerbosePrinter(r0, indent=2*' ', flush=True)
print_i4 = getVerbosePrinter(r0, indent=4*' ', flush=True)
#################################################


#################################################
def print_section(title, indent=0):
    _print('\n' + indent*' ' + '*** ' + title + ' ***')
#################################################


#################################################
def print_describe_content(s, f):
    _print('#***************************' + '*'*len(s) + '****', file=f)
    _print('#*** This file contains the ' + s          + ' ***', file=f)
    _print('#***************************' + '*'*len(s) + '****', file=f)
#################################################
    

##########################################################################
def print_matrix(m, maxcol=10):

    nrow = m.shape[0]
    ncol = m.shape[1]
    
    nblock = int(np.ceil(ncol/maxcol))
    j1 = 0
    j2 = min(maxcol, ncol) - 1
    for b in range(0, nblock):

        #== Column title ==#
        _print('%5s   ' % '', end='')
        for j in range(j1, j2+1):
            _print('%11d' % (j+1), end='')
        _print('')
        if b < nblock-1:
            hline = ''.join(['-' for i in range(0, 11*maxcol)])
        elif b == nblock-1:
            hline = ''.join(['-' for i in range(0, 11*(j2-j1+1))])
        _print('%5s   ' % '', end='')
        _print(hline)

        #== Population values ==#
        for i in range(0, nrow):
            _print('%5d  |' % (i+1), end='')
            for j in range(j1, j2+1):
                _print('%11.6f' % m[i,j], end='')
            _print('')
        _print('')
        j1 += maxcol
        j2 = min(j1+maxcol, ncol) - 1

##########################################################################


##########################################################################
def print_orb_occupations(occs):

    assert len(occs.shape) == 2
    assert occs.shape[0] == 2

    maxcol = 5
    print_section('Molecular orbitals occupations (alpha, beta)', 2)
    print_i4('', end='')
    for i in range(0, occs.shape[1]):
        _print('(%d: %8.6f, %8.6f)' % (i+1, occs[0,i], occs[1,i]), end='  ')
        if (i+1)%maxcol == 0 and i < occs.shape[1]-1:
            _print('')
            print_i4('', end='')
    _print('')

##########################################################################


##########################################################################
def print_partial_charge(mol, qmul, qlow):

    print_section('Atomic Mulliken populations', 2)
    print_i4('', end='')
    for i in range(0, mol.natm):
        atom = mol.atom_symbol(i) + str(i+1)
        _print('(%s: %.6f)' % (atom, qmul[i]), end='  ')
        if (i+1)%10 == 0 and i < mol.natm-1:
            _print('')
            print_i4('', end='')
    _print('')

    print_section('Atomic Lowdin populations', 2)
    print_i4('', end='')
    for i in range(0, mol.natm):
        atom = mol.atom_symbol(i) + str(i+1)
        _print('(%s: %.6f)' % (atom, qlow[i]), end='  ')
        if (i+1)%10 == 0 and i < mol.natm-1:
            _print('')
            print_i4('', end='')
    _print('')
##########################################################################


##########################################################################
def print_multipole(e_dpole, n_dpole, e_qpole, n_qpole):

    assert e_dpole.shape == (3,)
    assert n_dpole.shape == (3,)
    assert e_qpole.shape == (3,3)
    assert n_qpole.shape == (3,3)

    print_section('Multipole moment components', 2)
    mp_s = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz']
    print_i4('%10s' % '', end='  ')
    for i in range(0, 3): _print('%11s' % (6*' '+mp_s[i]+4*' '), end=' ')
    for i in range(3, 9): _print('%11s' % (6*' '+mp_s[i]+3*' '), end=' ')
    _print('')

    print_i4('%-10s' % 'Electronic', end='  ')
    for i in range(0, 3): _print('%11.6f' % e_dpole[i], end=' ')
    for i in range(0, 3): _print('%11.6f' % np.diag(e_qpole,0)[i], end=' ')
    for i in range(0, 2): _print('%11.6f' % np.diag(e_qpole,1)[i], end=' ')
    for i in range(0, 1): _print('%11.6f' % np.diag(e_qpole,2)[i])

    print_i4('%-10s' % 'Nuclear', end='  ')
    for i in range(0, 3): _print('%11.6f' % n_dpole[i], end=' ')
    for i in range(0, 3): _print('%11.6f' % np.diag(n_qpole,0)[i], end=' ')
    for i in range(0, 2): _print('%11.6f' % np.diag(n_qpole,1)[i], end=' ')
    for i in range(0, 1): _print('%11.6f' % np.diag(n_qpole,2)[i])

    print_i4('%-10s' % 'Total', end='  ')
    dpole = e_dpole + n_dpole
    qpole = e_qpole + n_qpole
    for i in range(0, 3): _print('%11.6f' % dpole[i], end=' ')
    for i in range(0, 3): _print('%11.6f' % np.diag(qpole,0)[i], end=' ')
    for i in range(0, 2): _print('%11.6f' % np.diag(qpole,1)[i], end=' ')
    for i in range(0, 1): _print('%11.6f' % np.diag(qpole,2)[i])
##########################################################################
