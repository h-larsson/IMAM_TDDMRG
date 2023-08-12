import numpy as np
from scipy.fft import rfft
from scipy.special import erfc


######################################
#=======================#
#==== USAGE EXAMPLE ====#
#=======================#

'''
from IMAM_TDDMRG.observables.fourier import fft_pcharge

atoms = ['H1', 'C2', 'C3', 'Cl4']
prefix = 'Acetylene-chloro'
fft_dir = '.'
t_path = './Acetylene-chloro.ts.npy'
q_path = './Acetylene-chloro.low.npy'
pad_t = True
pad_factor = 5
smh_end = True
smh_t0 = 0.15
smh_dt = 10.0

fft_pcharge.fft(atoms, q_path, t_path,  fft_dir, prefix, pad_t, pad_factor,
                smh_end=smh_end, smh_dt=smh_dt, smh_t0=smh_t0, header_lines=7)

'''
######################################








######################################################
def fft(atoms, q_path, t_path=None, fft_dir='.', prefix='', pad_t=False, pad_factor=3,
        smh_end=False, smh_dt=1.0, smh_t0=0.0, header_lines=7):
    '''
    DESCRIPTION:
       This function calculates the Fourier transform of the time-dependent partial charge
       and then print it to a text file. Before the transform is performed, the value of 
       the time domain data will first be shifted so that the value at the final time is 
       zero.
    
    
    atoms:
       A list containing the atomic symbols as strings that will be used as the column 
       titles. No consistency checks are performed to these strings, hence they can 
       differ from the actual atomic symbols in the main simulation without throwing any
       error messages.
    
    q_path: 
       The (relative or full) path of the file containing the partial charge data
       to be Fourier transformed. The extension of this file must be 'npy' or 'low'. 
       In the first case, this function will load the time domain partial charge 
       from the '*.npy' file produced by the main simulation. t_path must also be 
       given the path of the '*.npy' file containing the sampling time data that is
       also produced by the main simulation. In the second case, i.e. when the 
       extension is 'low', the partial charge data will be loaded from the '*.low'
       text file produced by the main simulation. In this case, t_path will not be 
       used because the sampling times will be extracted from the '*low' file. Note 
       that when loading from a '*.low' file, the limit of 100000 lines of data in
       this file applies.
       
    t_path: 
       The path to the *.npy file containing the sampling time data. Required when 
       reading from a '*.npy' file. Unused when reading from a '*.low' file.
    
    fft_dir:
       The output directory where the Fourier spectrum will be stored.
    
    prefix:
       The prefix of the file name containing the Fourier spectrum. The output file
       name will be '<prefix>.fft_low'.
    
    pad_t:
       If True, then the shifted time domain data will be padded with zero.
    
    pad_factor:
       An integer that determines the padding length. The length of the padding is 
       given by 
          pad_factor*nt 
       where nt is the length of the unpadded partial charge data.       
    
    smh_end:
       If True, then the time domain data will be masked with an error-type function.
       The purpose of this masking is to smoothen the data at the final sampling time,
       which otherwise would probably have an abrupt drop to zero. The mask function 
       takes the form of 
          0.5 * erfc((t-t0)/smh_dt)
       where 
          t0 = delta_t * (1 - smh_t0)
          delta_t = the length of the unpadded sampling time, i.e. the difference
                    between the first and last element of the sampling time array.
    
    smh_dt:
       The length of the transition region of the smoothening function. See the 
       equation in the description of 'smooth_end' argument.
    
    smh_t0:
       The center of the smoothening error function. See the equation in the 
       description of 'smooth_end' argument.
    
    header_lines: 
       Only meaningful when reading from a '*.low' file. The number of header lines, 
       that is, the number of lines before the actual time domain data starts. 
    '''

    #==== Load the time and partial charge data ====#
    if q_path[-4 : len(q_path)] == '.npy':
        assert t_path is not None
        t = np.load(t_path)
        qt = np.load(q_path).real
        natm = qt.shape[0]
        assert natm == len(atoms)
    elif q_path[-4 : len(q_path)] == '.low':
        qtf = open(q_path, 'r')
        natm = len(atoms)
        t = []
        qt = []
        #== Loop over lines ==#
        for i in range(0,100000):
            s = qtf.readline().split()
            if s == []: break
            if i >= header_lines:
                t = t + [float(s[1])]
                qt_ = []
                for j in range(2, len(s)):
                    qt_ = qt_ + [float(s[j])]
                qt = qt + [ np.array(qt_) ]
            if i == 100000-1: print('WARNING: The limit of time point is reached. Some data ' +
                                    'may have been ignored.')
        t = np.array(t)
        qt = np.vstack(qt).T
    else:
        raise ValueError('fft_pcharge.fft: The file pointed to by q_path must have the ' +
                         'extension of either \'low\' or \'npy\'.')

    #==== Determine the time interval and length ====#
    print('The number of atoms = %d' % natm)
    nt = len(t)
    dt = t[1] - t[0]
    print('The number of time points = %d' % nt)
    print('Time interval = %.6e a.u. of time' % dt)

    #==== Apply the smoothening mask function ====#
    if smh_end:
        assert smh_t0 >= 0.0 and smh_t0 <= 1.0
        t0 = (t[-1] - t[0]) * (1-smh_t0)
        mask = np.array(0.5 * erfc((t - t0)/smh_dt))
        print('max. mask = %.4f,   min. mask = %.4f' % (max(mask), min(mask)))
    else:
        mask = 1.0
    for i in range(0, natm):
        qt[i,:] = (qt[i,:] - qt[i,-1]) * mask

    #==== Apply padding ====#
    if pad_t:
        npad = int(pad_factor * nt)
        nt = nt + npad
        print('The number of padded time points = %d' % nt)
        qt_ = np.zeros((natm,nt))
        for i in range(0, natm):
            qt_[i,:] = np.hstack( (qt[i,:], qt[i,-1]*np.ones(npad)) )
        qt = qt_

    #==== Determine the frequency domain interval and length ====#
    au2ev = 27.2113860200
    nw = int((nt/2)+1) if nt%2==0 else int((nt+1)/2)     # Based on the 'Returns' section of the scipy.fft.rfft? page.
    dw = 2*np.pi / (dt*nt) * au2ev
    w = np.linspace(0, (nw-1)*dw, num=nw)
    print('The number of omega points = %d' % nw)
    print('Omega interval = %.6e eV' % dw)

    #==== Compute the Fourier spectra ====#
    qw = np.zeros((natm, nw), dtype=np.complex128)
    for i in range(0, natm):
        qw[i,:] = rfft(qt)

    #==== Print the Fourier spectra ====#
    fft_file = fft_dir + '/' + prefix + '.fft_low' 
    with open(fft_file, 'w') as qwf:
        qwf.write('# 1 eV ' +
                  '= 1.5192674 rad/fs ' +
                  '= 0.0367493 rad/a.u. of time ' +
                  '= 241.79892 THz \n')
        
        qwf.write('%1s %5s  %11s' % ('#', 'No.', 'Omega (eV)'))
        for j in range(0, natm):
            qwf.write('  %14s' % (atoms[j]))
        qwf.write('\n')
        
        for i in range(0, nw):
            qwf.write('%1s %5d  %11.6f' % ('', i, w[i]))
            for j in range(0, natm):
                qwf.write('  %14.6e' % np.abs(qw[j,i]))
            qwf.write('\n')

    print('The Fourier transform of the Lowdin population loaded from ')
    print('  ' + q_path)
    print('has been printed to ')
    print('  ' + fft_file)
######################################################

