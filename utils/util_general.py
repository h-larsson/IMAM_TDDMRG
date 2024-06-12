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
def extract_timing(fpath, tinit, dt, tmax=None, substep_thr=1E-10, tthr=1E-12,
                   dtthr=1E-10):

    tlast = None
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


#######################################################
def timestat(prefix, dt, dfiles, av1=None, av2=None, save_dir='.'):
    '''
    dfiles:
       A dictionary whose element is in the following format
          <D_string>: [tinit, path0, restart_path1, ...]
       where D_string is the string signifying the bond dimension, tinit is
       the initial time for this TDDMRG propagation, path0 is the output 
       file of the initial TDDMRG simuluation that starts from tinit, and
       restart_path<n> are the optional restart files if any or if desired.
    '''
    
    step_cost = dfiles.copy()
    for Ds in step_cost.keys():
        step_cost[Ds] = []
    
    NN = 0
    for Ds, fpaths in dfiles.items():
        tinit = fpaths[0]
        step_cost[Ds] = np.array([])
        for fpath in fpaths[1:]:
            print('\nD = ', Ds)
            print('   File = ', fpath)
            step_cost0, tmax = extract_timing(fpath, tinit, dt)
            print('')
            if len(step_cost0) > 0:
                print('   Longest step cost = %.3f' % np.max(step_cost0))
                print('   Shortest step cost = %.3f' % np.min(step_cost0))
            print('   Number of recorded time points = ', len(step_cost0))
            step_cost[Ds] = np.hstack( (step_cost[Ds], np.array(step_cost0)) )
            tinit = tmax + dt
    
        print('')
        print('Total number of recorded time points = ', len(step_cost[Ds]))
        print('Average cost per step = %.3f' % np.mean(step_cost[Ds]))
        print('Standard deviation = %.3f' % np.std(step_cost[Ds]))
        NN = max(NN, len(step_cost[Ds]))
                    
            
    if av1 is None: av1 = 0
    if av2 is None: av2 = NN
    if not os.path.isdir(save_dir): os.mkdir(save_dir)
    atiming_file = save_dir + '/' + prefix + '.atm'
    with open(atiming_file, 'w') as tmf:
        tmf.write(f'Statistics between the {av1:d}-th and {av2:d}-th data series:\n')
        
        # column title
        tmf.write('%11s  ' % 'D')
        for Ds in step_cost.keys():
            tmf.write('  %10s' % Ds)
        tmf.write('\n')
    
        # table content
        tmf.write('%11s  ' % 'Mean')
        for Ds in step_cost.keys():
            tmf.write('  %10.3f' % np.mean(step_cost[Ds][av1:av2]))
        tmf.write('\n')
        tmf.write('%11s  ' % 'Std. dev.')
        for Ds in step_cost.keys():
            tmf.write('  %10.3f' % np.std(step_cost[Ds][av1:av2]))
        tmf.write('\n')
        tmf.write('%11s  ' % 'Max')
        for Ds in step_cost.keys():
            tmf.write('  %10.3f' % np.max(step_cost[Ds][av1:av2]))
        tmf.write('\n')
        tmf.write('%11s  ' % 'Min')
        for Ds in step_cost.keys():
            tmf.write('  %10.3f' % np.min(step_cost[Ds][av1:av2]))
        tmf.write('\n')
        tmf.write('%11s  ' % 'ndata')
        for Ds in step_cost.keys():
            tmf.write('  %10i' % len(step_cost[Ds][av1:av2]))
        tmf.write('\n')
            
    
    timing_file = save_dir + '/' + prefix + '.tm'
    with open(timing_file, 'w') as tmf:
        # column title
        tmf.write('#%7s  ' % 'No.')
        for Ds in step_cost.keys():
            tmf.write('  %10s' % Ds)
        tmf.write('\n')
    
        # table content
        for i in range(NN):
            tmf.write('%8i  ' % i)
            for Ds in step_cost.keys():
                if i <= len(step_cost[Ds])-1:
                    tmf.write('  %10.3f' % step_cost[Ds][i])
                else:
                    tmf.write('  %10s' % ' ')
            tmf.write('\n')
#######################################################
