import glob
import numpy as np
from pyscf import symm, tools
from TDDMRG_CM.utils import util_atoms, util_print, util_general
from TDDMRG_CM.observables import extract_time
from TDDMRG_CM.phys_const import au2fs


EXT1 = '.tc'
EXT2 = '.tcs.cube'
EXT3 = '.tcd.cube'

########################################################
def calc(rdm, mol=None, orb=None, nCore=None, nCAS=None, corr_dm=False, logbook=None):
    
    '''
    Calculate static and dynamic correlation indices and optionally, correlation density 
    matrices (used for plotting the local correlations).

    Input
    -----
    mol:
      Mole object describing the molecule. The information about AO basis is contained here.
    rdm: 
      The active space block of one-particle reduced density matrix in orthonormal orbital 
      basis representation where these orbitals are given by orb. rdm should be
      spin-resolved, i.e. rdm[0,:,:] and rdm[1,:,:] must correspond to the RDMs in alpha 
      and beta spin-orbitals.
    orb:
      AO coefficients of active space orbitals in which rdm is represented.

    Output
    ------
    o_s:
      Natural spin-orbital contribution to the global static correlation index. To get
      the global static correlation index, simply sum over all of its elements.
    o_d:
      Same matrix as for o_s except that it is for dynamic correlation.
    corr_s:
      Static correlation density matrix, in 'AO reprsentation', which is the representation
      for any one-particle density matrix-like matrix that can directly be input to 
      pyscf.tools.cubegen.density function to plot its values in 3D space.
    corr_d:
      Same matrix as for corr_s except that it is for dynamic correlation.
    '''

    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]
        
    assert len(rdm.shape) == 3
    assert orb.shape[1] == rdm.shape[1], f'{orb.shape[1]} vs. {rdm.shape[1]}'
    cpx = isinstance(rdm[0,0,0], complex)
    if cpx:
        dtype = complex
    else:
        dtype = float

    #==== Compute natural occupancies and orbitals ====#
    osym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orb)
    natocc = np.zeros((2,rdm.shape[2]), dtype=dtype)
    natorb_ = np.zeros(rdm.shape, dtype=dtype)
    for i in range(2):
        natocc[i,:], natorb_[i,:,:] = symm.eigh(rdm[i,:,:], osym)
    natocc = natocc.real

    #==== Calculate static and dynamic correlation indices ====#
    o_s = np.zeros(natocc.shape)
    o_d = np.zeros(natocc.shape)
    for i in range(2):
        o_s[i,:] = 0.50 * natocc[i,:]*(1-natocc[i,:])
        o_d[i,:] = 0.25 * ( np.sqrt(natocc[i,:]*(1-natocc[i,:])) -
                            2*natocc[i,:]*(1-natocc[i,:]) )

    #==== Calculate static and dynamic correlation density matrices in pseudo-AO basis ====#
    if corr_dm:
        #==== Transform nat. orbitals from orb rep. to AO rep. ====#
        # natorb_ = nat. orbitals in input orbitals representation
        # natorb = nat. orbitals in AO representation
        natorb = np.zeros((2, mol.nao, natorb_.shape[2]), dtype=dtype)
        for i in range(2): natorb[i,:,:] = orb @ natorb_[i,:,:]
        
        corr_s = np.zeros((mol.nao,mol.nao), dtype=dtype)
        corr_d = np.zeros((mol.nao,mol.nao), dtype=dtype)
        for i in range(2):
            corr_s = corr_s + (natorb[i,:,:] * o_s[i,:]) @ natorb[i,:,:].conj().T     # natorb @ o_s @ natorb.H
            corr_d = corr_d + (natorb[i,:,:] * o_d[i,:]) @ natorb[i,:,:].conj().T     # natorb @ o_d @ natorb.H
    else:
        corr_s = corr_d = None

    return o_s, o_d, corr_s, corr_d
########################################################


########################################################
def td_calc(mol=None, tdir=None, orb=None, nCore=None, nCAS=None, nelCAS=None, 
            corr_dm=False, nc=(30,30,30), prefix='loc_corr', simtime_thr=1E-11, tnorm=True,
            verbose=2, logbook=None):
    '''
    orb:
      AO coefficients of active space orbitals in which rdm is represented.
    '''
    
    if mol is None:
        mol = util_atoms.mole(logbook)
    if nCore is None:
        nCore = logbook['nCore']
    if nCAS is None:
        nCAS = logbook['nCAS']
    if nelCAS is None:
        nelCAS = logbook['nelCAS']
    if orb is None:
        orb = np.load(logbook['orb_path'])[:,nCore:nCore+nCAS]
    if tdir is None:
        tdir = logbook['sample_dir']
    outname = prefix + EXT1

    #==== Construct the time array ====#
    tt, _, ntevo, rdm_dir = util_general.extract_tevo(tdir)
    idsort = np.argsort(tt)
    ndigit = len(str(ntevo))
    
    #==== Print column titles ====#
    with open(outname, 'w') as outf:
        outf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
        outf.write('#%9s %13s  ' % ('Col #1', 'Col #2'))
        outf.write(' %16s %16s' % ('Col #3', 'Col #4'))
        outf.write('\n')
        outf.write('#%9s %13s  ' % ('No.', 'Time (a.u.)'))
        outf.write(' %16s %16s' % ('Static id.', 'Dynamic id.'))
        outf.write('\n')

    k = 0
    kk = 0
    for i in idsort:
        if kk > 0:
            assert not (tt[i] < t_last), 'Time points are not properly sorted, this is ' \
                'a bug in the program. Report to the developer. ' + \
                f'Current time point = {tt[i]:13.8f}.'

        #==== Remove existing *.tdh files ====#
        oldfiles = glob.glob(rdm_dir[i] + '/*' + EXT2) + \
                   glob.glob(rdm_dir[i] + '/*' + EXT3)
        for f in oldfiles:
            os.remove(f)

        #==== Load cation RDM1 ====#
        rdm = np.load(rdm_dir[i] + '/1pdm.npy')
        tr = np.sum( np.trace(rdm, axis1=1, axis2=2) ).real
        if tnorm:
            rdm = rdm * nelCAS / tr
            
        #==== When the time point is different from the previous one ====#
        if (kk > 0 and tt[i]-t_last > simtime_thr) or kk == 0:
            if verbose > 1:
                print('%d) Time point: %.5f fs' % (k, tt[i]))
                print('    RDM1 loaded from ' + rdm_dir[i])
                print('    RDM trace = %.8f' % tr)
            echeck = np.linalg.eigvalsh(np.sum(rdm, axis=0))
            o_s, o_d, corr_s, corr_d = calc(rdm, mol, orb, nCore, nCAS, corr_dm)
            i_s = np.sum(o_s)
            i_d = np.sum(o_d)

            #==== Print correlation indices ====#
            with open(outname, 'a') as outf:
                #== Print time ==#
                outf.write(' %9d %13.8f  ' % (k, tt[i]))
    
                #== Print correlation indices ==#
                outf.write(' %16.6e %16.6e' % (i_s, i_d))
                outf.write('\n')

            #==== Cube-print local correlation functions ====#
            if corr_dm:
                assert corr_s is not None and corr_d is not None
                cubename = rdm_dir[i] + '/' + prefix + '-' + str(k).zfill(ndigit) + EXT2
                if verbose > 1:
                    print('    Printing static local correlation into ' + cubename)
                tools.cubegen.density(mol, cubename, corr_s, nx=nc[0], ny=nc[1], nz=nc[2])
                cubename = rdm_dir[i] + '/' + prefix + '-' + str(k).zfill(ndigit) + EXT3
                if verbose > 1:
                    print('    Printing dynamic local correlation into ' + cubename)
                tools.cubegen.density(mol, cubename, corr_d, nx=nc[0], ny=nc[1], nz=nc[2])

            #==== Increment unique time point index ====#
            k += 1

        #==== When the time point is similar to the previous one ====#
        elif kk > 0 and tt[i]-t_last < simtime_thr:
            util_print.print_warning\
                ('The data loaded from \n    ' + rdm_dir[i] + '\nhas a time point almost ' +
                 'identical to the previous time point. Duplicate time point = ' +
                 f'{tt[i]:13.8f}')
            echeck_tsim = np.linalg.eigvalsh(np.sum(rdm, axis=0))
            max_error = max(np.abs(echeck_tsim - echeck))
            if max_error > 1E-6:
                util_print.print_warning\
                    (f'The 1RDM loaded at the identical time point {tt[i]:13.8f} yields ' +
                     f'eigenvalues different by up to {max_error:.6e} as the other identical \n' +
                     'time point. Ideally you don\'t want to have such inconcsistency ' +
                     'in your data. Proceed at your own risk.')
            else:
                print('   Data at this identical time point is consistent with the previous ' + \
                      'time point. This is good.\n')

        t_last = tt[i]

        #==== Increment general (non-unique) time point index ====#
        kk += 1
########################################################
