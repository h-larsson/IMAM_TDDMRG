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
