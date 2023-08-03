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
def print_partial_charge(mol, qmul, qlow):

    print_section('Mulliken populations', 2)
    print_i4('', end='')
    for i in range(0, mol.natm):
        atom = mol.atom_symbol(i) + str(i+1)
        _print('(%s: %.6f)' % (atom, qmul[i]), end='  ')
        if (i+1)%10 and i < mol.natm-1 == 0:
            _print('')
            print_i4('', end='')
    _print('')

    print_section('Lowdin populations', 2)
    print_i4('', end='')
    for i in range(0, mol.natm):
        atom = mol.atom_symbol(i) + str(i+1)
        _print('(%s: %.6f)' % (atom, qlow[i]), end='  ')
        if (i+1)%10 and i < mol.natm-1 == 0:
            _print('')
            print_i4('', end='')
    _print('')
##########################################################################
