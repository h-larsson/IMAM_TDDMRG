import numpy as np
from scipy.special import erfc
from scipy.fft import rfft


######################################
#=======================#
#==== USAGE EXAMPLE ====#
#=======================#
'''
from IMAM_TDDMRG.observables.fourier import fft_ac

prefix = 'Acetylene-chloro'
fft_dir = '.'
ac_path = './Acetylene-chloro.ac'
pad_t = True
factor = 5
smh_end = True
smh_t0 = 0.15
smh_dt = 10.0


fft_ac.fft(ac_path, fft_dir, prefix, 0.05, pad_t=pad_t, pad_factor=factor,
           smh_end=smh_end, smh_t0=smh_t0, smh_dt=smh_dt)
'''
######################################








######################################################
def fft(ac_path, fft_dir='.', prefix='', dt=None, dt_tol=1.0E-12, pad_t=False, pad_factor=3,
        smh_end=False, smh_dt=1.0, smh_t0=0.0, header_lines=7):
    '''
    DESCRIPTION:
       This function calculates the Fourier transform of the time-dependent autocorrelation
       and then print it to a text file. Before the transform is performed, the value of 
       the time domain data will first be shifted so that the value at the final time is 
       zero.
    
        
    ac_path: 
       The (relative or full) path of the file containing the autocorrelation data
       to be Fourier transformed. 
           
    fft_dir:
       The output directory where the Fourier spectrum will be stored.
    
    prefix:
       The prefix of the file name containing the Fourier spectrum. The output file
       name will be '<prefix>.fft_ac'.

    dt:
       The time interval in a.u. of time at which the autocorrelation data will be 
       read from ac_path. Useful when the time-dependent autocorrelation is sampled
       at irregularly spaced time points to extract only the regularly spaced 
       subset.
    
    dt_tol:
       The tolerance that will be used in conjunction with dt to extract the 
       regularly spaced subset of the autocorrelation data.
    
    pad_t:
       If True, then the shifted time domain data will be padded with zero.
    
    pad_factor:
       An integer that determines the padding length. The length of the padding is 
       given by 
          pad_factor*nt 
       where nt is the length of the unpadded autocorrelation data.       
    
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
       The number of header lines, that is, the number of lines before the actual 
       time domain data starts.
    '''



    #==== Load the time and autocorrelation data ====#
    acf = open(ac_path, 'r')
    t = []
    at = []
    #== Loop over lines ==#
    for i in range(0,100000):
        s = acf.readline().split()
        if s == []: break
        if i == header_lines:
            t = t + [float(s[1])]
            at = at + [float(s[4])]
        elif i > header_lines:
            t_ = float(s[1]) 
            if dt is not None and abs((t_ - t[-1]) - dt) < dt_tol:
                t = t + [t_]
                at = at + [float(s[4])]
        if i == 100000-1: print('WARNING: The limit of time point is reached. Some data ' +
                                'may have been ignored.')

    #==== Apply the smoothening mask function ====#
    t = np.array(t)
    at = np.array(at)
    if smh_end:
        assert smh_t0 >= 0.0 and smh_t0 <= 1.0
        t0 = (t[-1] - t[0]) * (1-smh_t0)
        mask = np.array(0.5 * erfc((t - t0)/smh_dt))
        print('max. mask = %.4f,   min. mask = %.4f' % (max(mask), min(mask)))
    else:
        mask = 1.0
    at = (at - at[-1]) * mask

    #==== Determine the time interval and length ====#
    nt = len(t)
    dt = t[1] - t[0]
    print('The number of time points = %d' % nt)
    print('Time interval = %.6e a.u. of time' % dt)

    #==== Apply padding ====#
    if pad_t:
        npad = int(pad_factor * nt)
        nt = nt + npad
        print('The number of padded time points = %d' % nt)
        at = np.hstack( (at, at[-1]*np.ones(npad)) )

    #==== Determine the frequency domain interval and length ====#
    au2ev = 27.2113860200
    nw = int((nt/2)+1) if nt%2==0 else int((nt+1)/2)     # Based on the 'Returns' section of the scipy.fft.rfft? page.
    dw = 2*np.pi / (dt*nt) * au2ev
    w = np.linspace(0, (nw-1)*dw, num=nw)
    print('The number of omega points = %d' % nw)
    print('Omega interval = %.6e eV' % dw)
    aw = rfft(at)

    #==== Print the Fourier spectra ====#
    fft_file = fft_dir + '/' + prefix + '.fft_ac' 
    with open(fft_file, 'w') as awf:
        awf.write('# 1 eV ' +
                  '= 1.5192674 rad/fs ' +
                  '= 0.0367493 rad/a.u. of time ' +
                  '= 241.79892 THz \n')
        
        awf.write('%1s %5s  %11s' % ('#', 'No.', 'Omega (eV)'))
        awf.write('  %14s' % 'FT Autocorr.')
        awf.write('\n')
        
        for i in range(0, nw):
            awf.write('%1s %5d  %11.6f' % ('', i, w[i]))
            awf.write('  %14.6e' % np.abs(aw[i]))
            awf.write('\n')

    print('The Fourier transform of the autocorrelation loaded from ')
    print('  ' + ac_path)
    print('has been printed to ')
    print('  ' + fft_file)
######################################################

