import numpy as np
from scipy.special import erfc
from TDDMRG_CM.phys_const import au2fs


def mask(mtype, t, par1=1.0, par2=0.0, verbose=True):
    if mtype == 'cos':
        assert isinstance(par1, int)
        delta_t = t[-1] - t[0]
        mask = (np.cos(np.pi*t/2/delta_t))**par1 * np.heaviside(1-abs(t)/delta_t, 0.5)
        if verbose:
            print('Masking is active')
            print('  type = cos')
            print('  max. mask = %.4f,   min. mask = %.4f' % (max(mask), min(mask)))
            print('  cosine power = %d' % par1)
    elif mtype == 'erfc':
        assert par2 >= 0.0 and par2 <= 1.0
        t0 = (t[-1] - t[0]) * (1-par2)
        mask = np.array(0.5 * erfc((t - t0)/par1))
        if verbose:
            print('Masking is active')
            print('  type = erfc')
            print('  max. mask = %.4f,   min. mask = %.4f' % (max(mask), min(mask)))
            print('  damping center = %.6e a.u. = %.6e fs' % (t0, t0*au2fs))
            print('  damping duration = %.6e a.u. = %.6e fs' % (2*par1, 2*par1*au2fs))
    else:
        mask = np.ones(len(t))
        if verbose:
            print('Masking not active')

    return mask
