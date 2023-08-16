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

from IMAM_TDDMRG.phys_const import au2fs
    


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
def print_warning(s):
    _print('\n\n')
    _print('>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<')
    _print('>>>> WARNING <<>> WARNING <<>> WARNING <<<<')
    _print('>>>>>>>>>>>>>>>>>>>>>-<<<<<<<<<<<<<<<<<<<<<')
    print_i2(s)
    _print('\n\n')
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
def print_pcharge(mol, qmul, qlow):

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
def print_mpole(e_dpole, n_dpole, e_qpole, n_qpole):

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


##########################################################################
class print_autocorrelation:

    #################################################
    def __init__(self, prefix, n_t, save_txt=True, save_npy=True):
        self.it = -1
        self.prefix = prefix
        self.ac_f = './' + self.prefix + '.ac'
        self.ac2t_f = './' + self.prefix + '.ac2t'
        self.ac = np.zeros(n_t, dtype=np.complex128)
        self.ac2t = np.zeros(n_t, dtype=np.complex128)
        self.save_txt = save_txt
        self.save_npy = save_npy

        if not self.save_txt and not self.save_npy:
            print_warning('An object of print_autocorrelation class is initiated but is ' +
                          'set to print nothing.')
        
        self.header_stat = False
        self.print_stat = False
    #################################################


    #################################################
    def header(self):
        if self.save_txt:
            assert self.header_stat == False, \
                'Cannot double print the header of the autocorrelation files.'
            assert self.print_stat == False, \
                'Cannot print the header of the autocorrelation files while it is ' + \
                'printing the autocorrelation values.'

        hline = ''.join(['-' for i in range(0, 73)])
        
        #==== Initiate autocorrelation file ====#
        with open(self.ac_f, 'w') as acf:
            print_describe_content('autocerrelation data', acf)
            acf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
            acf.write('#' + hline + '\n')
            acf.write('#%9s %13s   %11s %11s %11s %11s\n' %
                      ('No.', 'Time (a.u.)', 'Real part', 'Imag. part', 'Abs', 'Norm'))
            acf.write('#' + hline + '\n')

        #==== Initiate 2t autocorrelation file ====#
        with open(self.ac2t_f, 'w') as ac2tf:
            print_describe_content('2t-autocerrelation data', ac2tf)
            ac2tf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
            ac2tf.write('#' + hline + '\n')
            ac2tf.write('#%9s %13s   %11s %11s %11s\n' %
                        ('No.', 'Time (a.u.)', 'Real part', 'Imag. part', 'Abs'))
            ac2tf.write('#' + hline + '\n')

        self.header_stat = True
    #################################################


    #################################################
    def print_ac(self, tt, ac, ac2t, normsqs):
        if self.save_txt:
            assert self.header_stat == True, \
                'The header of the autocorrelation files must be printed first (by ' + \
                'calling print_autocorrelation.header() before printing the ' + \
                'autocorrelation values.'
        if self.save_txt or self.save_npy:
            assert isinstance(ac, (complex, np.complex64, np.complex128))
            assert isinstance(ac2t, (complex, np.complex64, np.complex128))
            
        self.it += 1
        self.ac[self.it] = ac
        self.ac2t[self.it] = ac2t
        
        if self.save_txt:
            with open(self.ac_f, 'a') as acf:
                acf.write(' %9d %13.8f   %11.8f %11.8f %11.8f %11.8f\n' %
                          (self.it, tt, self.ac[self.it].real, self.ac[self.it].imag, 
                           np.abs(self.ac[self.it]), normsqs) )
            with open(self.ac2t_f, 'a') as ac2tf:
                ac2tf.write(' %9d %13.8f   %11.8f %11.8f %11.8f\n' %
                            (self.it, 2*tt, self.ac2t[self.it].real, self.ac2t[self.it].imag,
                             np.abs(self.ac2t[self.it])) )
            self.print_stat = True

        if self.save_npy:
            np.save(self.ac_f, self.ac[0:self.it+1])
            np.save(self.ac2t_f, self.ac2t[0:self.it+1])

##########################################################################


##########################################################################
class print_td_pcharge:

    #############################################
    def __init__(self, atom_symbol, prefix, n_t, max_atom=8, save_txt=True, save_npy=True):
        self.atom_symbol = atom_symbol
        self.natm = len(atom_symbol)
        self.prefix = prefix
        self.max_atom = max_atom
        self.nparts = int(np.ceil(self.natm / self.max_atom))
        self.rem = self.natm % self.max_atom
        self.save_txt = save_txt
        self.save_npy = save_npy
        self.low_file = []
        self.it = -1
        self.pchg = np.zeros((self.natm, n_t), dtype=np.complex128)

        if not self.save_txt and not self.save_npy:
            print_warning('An object of print_td_pcharge class is initiated but is set' +
                          'to print nothing.')
        
        self.header_stat = False
        self.print_stat = False
        self.footer_stat = False
    #############################################


    #############################################
    def header(self):
        if self.save_txt:
            assert self.header_stat == False, \
                'Cannot double print the header of the TD partial charge files.'
            assert self.print_stat == False, \
                'Cannot print the header of the TD partial charge files while it is ' + \
                'printing the partial charge values.'
            assert self.footer_stat == False, \
                'Cannot print the header of the TD partial charge files when the footer ' + \
                'has been printed as this indicates the end of the TD partial charge ' + \
                'printing.'

        
        #==== Loop over files ====#
        if self.save_txt:
            ia = 0
            for i in range(0, self.nparts):
                #==== File names ====#
                self.low_file += ['./' + self.prefix + '.' + str(i+1) + '.low']
            
                #==== ncol = the no. of columns for the current file ====#
                ncol = self.max_atom
                if i == self.nparts-1 and self.rem != 0:
                    ncol = self.rem
                hline = ''.join(['-' for i in range(0, 9+1+13+2+(1+14)*ncol)])
            
                #==== Begin printing the colum title ====#
                with open(self.low_file[i], 'w') as lowf:
                    print_describe_content('Lowdin partial charge data', lowf)
                    lowf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
                    lowf.write('#' + hline + '\n')
                    lowf.write('#%9s %13s  ' % ('No.', 'Time (a.u.)'))
            
                    #== Loop over atoms for the current file ==#
                    for j in range(0, ncol):
                        lowf.write(' %14s' % (self.atom_symbol[ia] + str(ia+1)))
                        ia += 1
                    lowf.write('\n')
                    lowf.write('#' + hline + '\n')
            
            self.header_stat = True
    #############################################


    #############################################
    def print_pcharge(self, tt, pchg):
        if self.save_txt:
            assert self.header_stat == True, \
                'The header of the TD partial charge files must be printed first (by ' + \
                'calling print_td_pcharge.header() before printing the partial charge ' + \
                'values.'
            assert self.footer_stat == False, \
                'Cannot print the values of the TD partial charge when the footer has ' + \
                'been printed as this indicates the end of the TD partial charge printing.'
        if self.save_txt or self.save_npy:
            assert isinstance(pchg[0], (np.complex64, np.complex128))
            assert len(pchg) == self.natm, \
                f'len(pchg) = {len(pchg)} while self.natm = {self.natm}.'
        
        self.it += 1
        self.pchg[:,self.it] = pchg

        #==== Printing into text files ====#
        if self.save_txt:
            ia = 0

            #==== Loop over files ====#
            for i in range(0, self.nparts):
                ncol = self.max_atom
                if i == self.nparts-1 and self.rem != 0:
                    ncol = self.rem
                with open(self.low_file[i], 'a') as lowf:
                    lowf.write(' %9d %13.8f  ' % (self.it, tt))

                    #== Loop over atoms for the current file ==#
                    for j in range(0, ncol):
                        lowf.write(' %14.6e' % (self.pchg[ia,self.it].real))
                        ia += 1
                    lowf.write('\n')
                    
            self.print_stat = True

                    
        #==== Printing into *.npy file ====#
        if self.save_npy:
            npy_file = './' + self.prefix + '.low'
            np.save(npy_file, self.pchg[:, 0:self.it+1])
    #############################################


    #############################################
    def footer(self):
        if self.save_txt:
            assert self.header_stat == True, \
                'The header of the TD partial charge files must be printed first (by ' + \
                'calling print_td_pcharge.header() before printing the footer.'
            assert self.footer_stat == False, \
                'Cannot double print the footer of the TD partial charge files.'
        
        im_max = np.max(self.pchg.imag, axis=1)
        im_min = np.min(self.pchg.imag, axis=1)

        #==== Loop over files ====#
        if self.save_txt:
            ia = 0
            for i in range(0, self.nparts):
                with open(self.low_file[i], 'a') as lowf:
                    lowf.write('\n')
                    if self.print_stat == False:
                        lowf.write('# WARNING: No partial charge values have been printed.\n')
                        
                    lowf.write('# Statistics of the imaginary parts (max,min): \n')
                    ncol = self.max_atom
                    if i == self.nparts-1 and self.rem != 0:
                        ncol = self.rem
                        
                    #== Loop over the atoms for the current file ==#
                    for j in range(0, ncol):
                        lowf.write('#  %s: %17.8e, %17.8e \n' %
                                   (self.atom_symbol[ia] + str(ia+1),
                                    im_max[ia], im_min[ia]))
                        ia += 1
    
            self.footer_stat = True
##########################################################################


##########################################################################
class print_td_mpole:

    #############################################
    def __init__(self, prefix, n_t, save_txt=True, save_npy=True):
        self.prefix = prefix
        self.save_txt = save_txt
        self.save_npy = save_npy
        self.mp_file = './' + self.prefix + '.mp'
        self.it = -1
        self.dp = np.zeros((3, n_t), dtype=np.complex128)    # Dipole moments
        self.qp = np.zeros((6, n_t), dtype=np.complex128)    # Quadrupole moments

        if not self.save_txt and not self.save_npy:
            print_warning('An object of print_td_mpole class is initiated but is set' +
                          'to print nothing.')
            
        self.header_stat = False
        self.print_stat = False
        self.footer_stat = False
    #############################################


    #############################################
    def header(self):
        if self.save_txt:
            assert self.header_stat == False, \
                'Cannot double print the header of the TD multipole files.'
            assert self.print_stat == False, \
                'Cannot print the header of the TD multipole files while it is ' + \
                'printing the multipole values.'
            assert self.footer_stat == False, \
                'Cannot print the header of the TD multipole files when the footer ' + \
                'has been printed as this indicates the end of the TD multipole ' + \
                'printing.'

        if self.save_txt:
            hline = ''.join(['-' for i in range(0, 9+1+13+2+(1+14)*9)])
            with open(self.mp_file, 'w') as mpf:
                print_describe_content('multipole components data', mpf)
                mpf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
                mpf.write('#' + hline + '\n')
                mpf.write('#%9s %13s   %14s %14s %14s %14s %14s %14s %14s %14s %14s\n' %
                          ('No.', 'Time (a.u.)', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy',
                           'yz', 'xz'))
                mpf.write('#' + hline + '\n')
    
            self.header_stat = True
    #############################################


    #############################################
    def print_mpole(self, tt, e_dp, n_dp, e_qp, n_qp):
        if self.save_txt:
            assert self.header_stat == True, \
                'The header of the TD multipole files must be printed first (by calling ' + \
                'print_td_mpole.header() before printing the multipole values.'
            assert self.footer_stat == False, \
                'Cannot print the values of the TD multipole when the footer has been ' + \
                'printed as this indicates the end of the TD multipole printing.'
        if self.save_txt or self.save_npy:
            assert isinstance(e_dp[0], (np.complex64, np.complex128))
            assert isinstance(e_qp[0,0], (np.complex64, np.complex128))
            assert len(e_dp) == 3, f'len(e_dp) = {len(e_dp)} while it has to be 3.'
            assert len(n_dp) == 3, f'len(n_dp) = {len(n_dp)} while it has to be 3.'
            assert e_qp.shape == (3,3), \
                f'e_qp.shape = {e_qp.shape} while it has to be (3,3).'
            assert n_qp.shape == (3,3), \
                f'n_qp.shape = {n_qp.shape} while it has to be (3,3).'
        
        self.it += 1
        self.dp[:,self.it] = e_dp + n_dp
        qp_ = e_qp + n_qp
        self.qp[:,self.it] = np.hstack( (np.diag(qp_,0), np.diag(qp_,1), np.diag(qp_,2)) )

        #==== Printing into text files ====#
        if self.save_txt:
            with open(self.mp_file, 'a') as mpf:
                mpf.write((' %9d %13.8f  ' +
                           ' %14.6e %14.6e %14.6e' +
                           ' %14.6e %14.6e %14.6e' +
                           ' %14.6e %14.6e %14.6e\n') %
                          (self.it, tt,
                           self.dp[0,self.it].real, self.dp[1,self.it].real,
                           self.dp[2,self.it].real,
                           self.qp[0,self.it].real, self.qp[1,self.it].real,
                           self.qp[2,self.it].real,
                           self.qp[3,self.it].real, self.qp[4,self.it].real,
                           self.qp[5,self.it].real))
                
            self.print_stat = True

        #==== Printing into *.npy file ====#
        if self.save_npy:
            np.save(self.mp_file, np.vstack( (self.dp[:, 0:self.it+1],
                                              self.qp[:, 0:self.it+1]) ))

    #############################################


    #############################################
    def footer(self):
        if self.save_txt:
            assert self.header_stat == True, \
                'The header of the TD multipole files must be printed first (by calling ' + \
                'print_td_mpole.header() before printing the footer.'
            assert self.footer_stat == False, \
                'Cannot double print the footer of the TD multipole files.'
        
        dp_im_max = np.max(self.dp.imag, axis=1)
        dp_im_min = np.min(self.dp.imag, axis=1)
        qp_im_max = np.max(self.qp.imag, axis=1)
        qp_im_min = np.min(self.qp.imag, axis=1)

        if self.save_txt:
            with open(self.mp_file, 'a') as mpf:
                mpf.write('\n')
                mpf.write('# Statistics of the imaginary parts (max,min): \n')
                mpf.write('#   x: %17.8e, %17.8e\n' % (dp_im_max[0], dp_im_min[0]))
                mpf.write('#   y: %17.8e, %17.8e\n' % (dp_im_max[1], dp_im_min[1]))
                mpf.write('#   z: %17.8e, %17.8e\n' % (dp_im_max[2], dp_im_min[2]))
                mpf.write('#   xx: %17.8e, %17.8e\n' % (qp_im_max[0], qp_im_min[0]))
                mpf.write('#   yy: %17.8e, %17.8e\n' % (qp_im_max[1], qp_im_min[1]))
                mpf.write('#   zz: %17.8e, %17.8e\n' % (qp_im_max[2], qp_im_min[2]))
                mpf.write('#   xy: %17.8e, %17.8e\n' % (qp_im_max[3], qp_im_min[3]))
                mpf.write('#   yz: %17.8e, %17.8e\n' % (qp_im_max[4], qp_im_min[4]))
                mpf.write('#   xz: %17.8e, %17.8e\n' % (qp_im_max[5], qp_im_min[5]))
    
            self.footer_stat = True
##########################################################################
