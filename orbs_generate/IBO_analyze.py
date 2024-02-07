import os, glob
import numpy as np
import scipy.linalg
from pyscf import gto, scf, lo, tools
from IMAM_TDDMRG.utils import util_logbook, util_qm, util_print
from IMAM_TDDMRG.observables import extract_time
from IMAM_TDDMRG.orbs_generate import util_orbs, analyze_orbs



def analyze(inputs):
    assert isinstance(inputs, dict)
    
    #==== Preprocess inputs ====#
    if inputs['prev_logbook'] is not None:
        inputs = util_logbook.parse(inputs)     # Extract input values from an existing logbook if desired.
    print('\nInput parameters:')
    for kw in inputs:
        print('  ', kw, ' = ', inputs[kw])
    print(' ')
    
    #==== Calculate IAOs, IBOs, and IBOCs in AO basis ====#
    nOcc = inputs['nCore'] + inputs['nCAS']
    nCore = inputs['nCore']
    nCAS = inputs['nCAS']
    nelCAS = inputs['nelCAS']
    mol = gto.M(atom=inputs['inp_coordinates'], basis=inputs['inp_basis'], symmetry=inputs['inp_symmetry'])
    ovl = mol.intor('int1e_ovlp')
    ibo, iao = util_orbs.get_IBO(mol, align_groups=inputs.get('align_groups', None))
    iboc = util_orbs.get_IBOC(mol, boao=(ibo, iao))
    print('Number of AOs = ', mol.nao)
    print('Sizes of IAO, IBO, and IBOC = ', iao.shape[1], ibo.shape[1], iboc.shape[1])
    print('Trace of overlap matrix of IBO in AO rep. = %10.6f' % np.trace(ibo.T @ ovl @ ibo))
    print('Trace of overlap matrix of IBOC in AO rep. = %10.6f' % np.trace(iboc.T @ ovl @ iboc))
    print('Frobenius norm of overlap matrix between IBOC and IBO = %10.6f' % np.linalg.norm(iboc.T @ ovl @ ibo, ord='fro') )
    
    #==== Analyze IAOs ====#
    print('Analysis of the IAOs:')
    analyze_orbs.analyze(mol, iao)
    
    #==== Analyze IBOs ====#
    print('Analysis of the IBOs:')
    analyze_orbs.analyze(mol, ibo)
    
    #==== Analyze IBOC's ====#
    print('Analysis of the IBOC\'s:')
    analyze_orbs.analyze(mol, iboc)
    
    #==== Print IAOs, IBOs, and IBOCs into cube files ====#
    if inputs['print_IAO'] or inputs['print_IBO'] or inputs['print_IBOC']:
        ndigit_iao = len(str(iao.shape[1]))
        ndigit_ibo = len(str(ibo.shape[1]))
        ndigit_iboc = len(str(iboc.shape[1]))
        if not os.path.isdir(inputs['cube_dir']):
            os.mkdir(inputs['cube_dir'])
        for i in range(0, max(iao.shape[1], ibo.shape[1], iboc.shape[1])):
            if inputs['print_IAO'] and i < iao.shape[1]:
                cubename = inputs['cube_dir'] + '/iao-' + str(i+1).zfill(ndigit_iao) + '.cube'
                print(f'{i+1:d}) Printing cube files: ')
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, iao[:,i])
            if inputs['print_IBO'] and i < ibo.shape[1]:
                cubename = inputs['cube_dir'] + '/ibo-' + str(i+1).zfill(ndigit_ibo) + '.cube'
                print(f'{i+1:d}) Printing cube files: ')
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, ibo[:,i])
            if inputs['print_IBOC'] and i < iboc.shape[1]:
                cubename = inputs['cube_dir'] + '/iboc-' + str(i+1).zfill(ndigit_iboc) + '.cube'
                print(f'{i+1:d}) Printing cube files: ')
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, iboc[:,i])
    
    #==== Calculate IBOs in occupied orbitals basis ====#
    orbs = np.load(inputs['orb_path'])
    ibo_o = orbs.T @ ovl @ ibo
    n_ibo = ibo_o.shape[1]
    print('IBO trace = ', np.trace(ibo_o.T @ ibo_o))
    iboc_o = orbs.T @ ovl @ iboc
    n_iboc = iboc_o.shape[1]
    print('IBOC trace = ', np.trace(iboc_o.T @ iboc_o))
    
    #==== Construct the time array ====#
    if isinstance(inputs['tevo_dir'], list):
        tevo_dir = inputs['tevo_dir'].copy()
    elif isinstance(inputs['tevo_dir'], tuple):
        tevo_dir = inputs['tevo_dir']
    elif isinstance(inputs['tevo_dir'], str):
        tevo_dir = ( inputs['tevo_dir'] )
    tt = []
    for d in tevo_dir:
        tt = np.hstack( (tt, extract_time.get(d)) )
    idsort = np.argsort(tt)
    ntevo = len(tt)

    pdm_dir = []
    for d in tevo_dir:
        pdm_dir = pdm_dir + glob.glob(d + '/tevo-*')

    simtime_thr = inputs.get('simtime_thr', 1E-11)
    with open(inputs['out_file'], 'w') as ouf:
    
        #==== Print column numbers ====#
        ouf.write(' %9s %13s  ' % ('Col #1', 'Col #2'))
        for i in range(3, n_ibo + n_iboc + 3):
            ouf.write(' %16s' % ('Col #' + str(i)) )
        ouf.write('\n')
    
        #==== Print IBO and IBOC indices ====#
        ouf.write(' %9s %13s  ' % ('', ''))
        for i in range(0, n_ibo):
            ouf.write(' %16s' % ('IBO #' + str(i+1)) )
        for i in range(0, n_iboc):
            ouf.write(' %16s' % ('IBOC #' + str(i+1)) )
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
                print(str(k) + ')  t = ', tt[i], ' fs')
                print('     RDM path = ', pdm_dir[i])
                pdm_full = np.sum( util_qm.make_full_dm(nCore, pdm1), axis=0 )
                tr = np.trace(pdm_full[nCore:nOcc, nCore:nOcc])
                pdm_full[nCore:nOcc, nCore:nOcc] = pdm_full[nCore:nOcc, nCore:nOcc] *nelCAS / tr
                
                #== Calculate IBO and IBOC occupations ==#
                occ_ibo = np.diag(ibo_o[0:nOcc,:].T @ pdm_full @ ibo_o[0:nOcc,:]).real
                occ_iboc = np.diag(iboc_o[0:nOcc,:].T @ pdm_full @ iboc_o[0:nOcc:,:]).real
                print('     Sum IBO occ = ', np.sum(occ_ibo))
                print('     Sum IBOC occ = ', np.sum(occ_iboc))
                print('     Sum IBO+IBOC occ = ', np.sum(occ_ibo) + np.sum(occ_iboc))
                
                #== Print time ==#
                ouf.write(' %9d %13.8f  ' % (k, tt[i]))
                
                #== Print IBO and IBOC occupations ==#
                for j in range(0, ibo_o.shape[1]):
                    ouf.write(' %16.6e' % occ_ibo[j])
                for j in range(0, iboc_o.shape[1]):
                    ouf.write(' %16.6e' % occ_iboc[j])
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
            t_last = tt[i]
            kk += 1
