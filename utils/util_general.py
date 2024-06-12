import os, glob
import numpy as np
from IMAM_TDDMRG.observables import extract_time
from IMAM_TDDMRG.phys_const import rad2deg


#######################################################
def mkDir(folder: str):
    """ mkdir -p folder """
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass
#######################################################


#######################################################
def extract_tevo(sample_dirs, simtime_thr=1E-11):

    #==== Sample directories ====#
    if isinstance(sample_dirs, list):
        sdir = sample_dirs.copy()
    elif isinstance(sample_dirs, tuple):
        sdir = sample_dirs
    elif isinstance(sample_dirs, str):
        sdir = tuple( [sample_dirs] )

    #==== Time points ====#
    tt = []
    for d in sdir:
        tt = np.hstack( (tt, extract_time.get(d)) )     # tt is unsorted and may contain duplicate time points.

    #==== Unique time points ====#
    tsort = np.sort(tt)
    tu = [tsort[0]]       # tu is sorted and contain only unique time points.
    ntevo = 1
    for i in range(1,len(tsort)):
        if tsort[i]-tsort[i-1] > simtime_thr:
            tu = tu + tsort[i]
            ntevo += 1
    tu = np.array(tu)

    #==== Get 1RDM path names ====#
    tevo_dirs = []
    for d in sdir:
        tevo_dirs = tevo_dirs + glob.glob(d + '/tevo-*')
    
    return tt, tu, ntevo, tevo_dirs
#######################################################


#######################################################
def rotmat(a, b, c):

    A, B, C = a/rad2deg, b/rad2deg, c/rad2deg
    
    rota = np.array( [[ np.cos(A), -np.sin(A),        0.0],
                      [ np.sin(A),  np.cos(A),        0.0],
                      [       0.0,        0.0,        1.0]] )
                                                
    rotb = np.array( [[ np.cos(B),        0.0,  np.sin(B)],
                      [       0.0,        1.0,        0.0],
                      [-np.sin(B),        0.0,  np.cos(B)]] )
                                                
    rotc = np.array( [[ np.cos(C), -np.sin(C),        0.0],
                      [ np.sin(C),  np.cos(C),        0.0],
                      [       0.0,        0.0,        1.0]] )

    return rota @ rotb @ rotc

#######################################################


#######################################################
def extract_timing(fpath, tinit, dt, tmax=None, tthr=1E-12, dtthr=1E-10):

    step_cost = []
    with open(fpath, 'r') as df:        
        time_iline = -1000
        iline = 0
        inrange = False
        dtmatch = False
        step_size = None
        tt_ = -1E4
        substep_cost = []
        for line in df:
            words = line.split()
            if len(words) >= 0:
                if 'Time point :' in line:
                    ic = int(words[3])
                    if inrange and dtmatch:
                        if sum(substep_cost) > substep_thr:
                            step_cost += [sum(substep_cost)]
                            tlast = tt
                            print('   Time, step size = %.8f  %.8f' % (tt, step_size))
                            print('   Sub step cost =', end='')
                            for tx in substep_cost: print('  %.3f' % tx, end='')
                            print('')
                            print('   Cost for this step = %.3f' % step_cost[-1])
                            substep_cost = []
                    print('Time point', ic)
                elif 'TD-PROPAGATION' in line:
                    tt = float(words[4])
                    if tmax is None:
                        inrange = tt >= tinit-tthr
                    else:
                        inrange = tt >= tinit-tthr and tt <= tmax+tthr
                    step_size = tt - tt_
                    dtmatch = abs(step_size-dt) < dtthr
                    tt_ = tt
                elif 'Time sweep =' in line and inrange:
                    tcost = float(words[3])
                    time_iline = iline
                elif  '| Tread' in line and iline == time_iline+1 and dtmatch and \
                      inrange:
                    substep_cost += [tcost]
            iline += 1
            
    return step_cost, tlast
#######################################################
