import glob
import numpy as np
from IMAM_TDDMRG.utils import util_atoms, util_qm, util_print
from IMAM_TDDMRG.observables import extract_time



def calc(outfile, orbx, mol=None, tdir=None, orb=None, nCore=None, nCAS=None, nelCAS=None,
         simtime_thr=1E-11, logbook=None):

    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])
    if tdir is None:
        tdir = logbook['sample_dir']
        

    #==== Some constants ====#
    nOcc = nCore + nCAS
    ovl = mol.intor('int1e_ovlp')

    #==== Calculate the desired orbitals in occupied orbitals basis ====#
    orb_o = orb.T @ ovl @ orbx
    norb = orbx.shape[1]
    print('Trace of orbital overlap matrix = ', np.trace(orb_o.T @ orb_o))
    
    #==== Construct the time array ====#
    if isinstance(tdir, list):
        tevo_dir = tdir.copy()
    elif isinstance(tdir, tuple):
        tevo_dir = tdir
    elif isinstance(tdir, str):
        tevo_dir = tuple( [tdir] )
    tt = []
    for d in tevo_dir:
        tt = np.hstack( (tt, extract_time.get(d)) )
    idsort = np.argsort(tt)
    ntevo = len(tt)

    #==== Get 1RDM path names ====#
    pdm_dir = []
    for d in tevo_dir:
        pdm_dir = pdm_dir + glob.glob(d + '/tevo-*')

    #==== Begin printing occupation numbers ====#
    with open(outfile, 'w') as ouf:
    
        #==== Print column numbers ====#
        ouf.write(' %9s %13s  ' % ('Col #1', 'Col #2'))
        for i in range(3, norb + 3):
            ouf.write(' %16s' % ('Col #' + str(i)) )
        ouf.write('\n')
    
        #==== Print orbital indices ====#
        ouf.write(' %9s %13s  ' % ('', ''))
        for i in range(0, norb):
            ouf.write(' %16s' % ('orb #' + str(i+1)) )
        ouf.write('\n')
        
        k = 0
        kk = 0
        for i in idsort:
            if kk > 0:
                assert not (tt[i] < t_last), 'Time points are not properly sorted, this is a bug in ' \
                    ' the program. Report to the developer. ' + f'Current time point = {tt[i]:13.8f}.'
            if (kk > 0 and tt[i]-t_last > simtime_thr) or kk == 0:
                #== Construct the full PDM ==#
                pdm1 = np.load(pdm_dir[i] + '/1pdm.npy')
                echeck = np.linalg.eigvalsh(np.sum(pdm1, axis=0))
                print(str(k) + ')  t = ', tt[i], ' fs', flush=True)
                print('     RDM path = ', pdm_dir[i])
                pdm_full = np.sum( util_qm.make_full_dm(nCore, pdm1), axis=0 )
                tr = np.trace(pdm_full[nCore:nOcc, nCore:nOcc])
                pdm_full[nCore:nOcc, nCore:nOcc] = pdm_full[nCore:nOcc, nCore:nOcc] * nelCAS / tr
                
                #== Calculate orbital occupations ==#
                occ_orb = np.diag(orb_o[0:nOcc,:].T @ pdm_full @ orb_o[0:nOcc,:]).real
                print('     Sum of orbital occupations = ', np.sum(occ_orb))
                
                #== Print time ==#
                ouf.write(' %9d %13.8f  ' % (k, tt[i]))
                
                #== Print orbital occupations ==#
                for j in range(0, orb_o.shape[1]):
                    ouf.write(' %16.6e' % occ_orb[j])
                ouf.write('\n')
                k += 1
            elif kk > 0 and tt[i]-t_last < simtime_thr:
                util_print.print_warning('The data loaded from \n    ' + pdm_dir[i] + '\nhas a time point almost ' +
                                         f'identical to the previous time point. Duplicate time point = {tt[i]:13.8f}')
                pdm1 = np.load(pdm_dir[i] + '/1pdm.npy')
                echeck_tsim = np.linalg.eigvalsh(np.sum(pdm1, axis=0))
                if max(np.abs(echeck_tsim - echeck)) > 1E-6:
                    util_print.print_warning(f'The 1RDM loaded at the identical time point {tt[i]:13.8f} yields ' +
                                             'eigenvalues different by more than 1E-6 as the other identical \n' +
                                             'time point. Ideally you don\'t want to have such inconcsistency ' +
                                             'in your data. Proceed at your own risk.')
                else:
                    print('   Data at this identical time point is consistent with the previous ' + \
                          'time point. This is good.\n')
            t_last = tt[i]
            kk += 1
