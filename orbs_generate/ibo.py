import os, glob
import numpy as np
import scipy.linalg
from pyscf import gto, scf, lo, tools, symm
from IMAM_TDDMRG.utils import util_logbook, util_qm, util_print, util_atoms
from IMAM_TDDMRG.utils import util_print as uprint
from IMAM_TDDMRG.observables import extract_time
from IMAM_TDDMRG.orbs_generate import util_orbs, analyze_orbs, local_orbs


##########################################################################
def get_IBO(mol=None, oiao=None, mo_ref=None, by_symm=False, align_groups=None,
            logbook=None):
    '''
    oiao:
       The orthogonalized IAO in AO basis.
    mo_ref:
       The orbitals acting as the polarization mold based on which the IBO is constructed. In the
       original IBO formulation, it is the Hartree-Fock canonical orbitals.
    align_groups:
       A tuple of tuples. Each inner tuple consists of 1-base IBO indices to be aligned with 
       mo_ref. This input is most likely only useful in linear molecules having orbitals higher 
       than sigma (e.g. pi, delta, etc) where the lobes of these orbitals may not be aligned with
       any Cartesian axes.
    '''

    if mol is None:
        mol = util_atoms.mole(logbook)
        
    if align_groups is not None:
        assert isinstance(align_groups, (list, tuple)), 'align_groups must be a tuple or ' + \
            'list which in turn contains lists or tuples.'

    ovl = mol.intor('int1e_ovlp')
    mf = scf.RHF(mol).run()
    if mo_ref is None:
        mo_ref = mf.mo_coeff[:,mf.mo_occ>0]
    
    #==== Obtain orthogonalized IAOs ====#
    if oiao is None:
        iao = lo.iao.iao(mol, mf.mo_coeff[:,mf.mo_occ>0])
        #OLD e, v = scipy.linalg.eigh(ovl)
        #OLD x_i = v @ np.diag( np.sqrt(e) ) @ v.T   # X^-1 for symmetric orthogonalization
        #OLD iao = x_i @ iao_
        oiao = lo.vec_lowdin(iao, ovl)
    
    #==== Calculate IBOs in AO basis ====#
    if by_symm:
        ibo = np.zeros(mo_ref.shape)
        refs = np.array( symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_ref) )
        for s in set(refs):
            ibo[:,refs == s] = lo.ibo.ibo(mol, mo_ref[:,refs == s], iaos=oiao)
    else:
        ibo = lo.ibo.ibo(mol, mo_ref, iaos=oiao)

    #==== Alignment ====#
    if align_groups is not None:
        # The task in this block is to determine mix_coef that satisfies:
        #     I = mo_ref.T @ ovl @ ibo_new
        # where
        #     ibo_new = ibo_old @ mix_coef
        # ibo_new is needed in linear molecules because the lobes of ibo_old are not necessarily aligned along
        # any Cartesian axes. The first equation holds under the assumption that the irrep orderings in mo_ref and
        # ibo_new are the same. Hence, the irrep ordering of ibo_new follows that of mo_ref. The columns of ibo_old
        # must be degenerate, the columns of mo_ref may not be degenerate but better are.
        for ik in align_groups:
            ip = tuple( [ik[j]-1 for j in range(0,len(ik))] )
            print('  -> Aligning IBOs:', ip, '(0-based)')
    
            #== Find the most suitable MOs for mo_ref ==#
            c = mo_ref.T @ ovl @ ibo[:,ip]
            idx = np.argmax(np.abs(c), axis=0)
            print('       Indices of the most suitable reference MOs:', idx, '(0-based)')
            mo_ref = mo_ref[:, idx]
    
            #== Compute the new (aligned) IBOs ==#
            mix_coef = np.linalg.inv(mo_ref.T @ ovl @ ibo[:,ip])
            ibo0 = ibo[:,ip] @ mix_coef
            norms = np.einsum('ij, jk, ki -> i', ibo0.T, ovl, ibo0)  # np.diag(ibo.T @ ovl @ ibo)
            ibo[:,ip] = ibo0 / np.sqrt(norms)

    return ibo
##########################################################################


##########################################################################
def get_IBOC(mol=None, oiao=None, ibo=None, mo_ref=None, loc='IBO', by_symm=False, 
             align_groups=None, ortho_thr=1E-8, logbook=None):
    '''
    This function calculates the set of vectors orthogonal to IBOs (calculated by get_IBO)
    that are also spanned by IAOs.

    Inputs
    ------
    oiao:
       The orthogonalized IAO in AO basis.
    ibo:
       IBO in AO basis.
    mo_ref:
       The orbitals acting as the polarization mold based on which the IBO is constructed. In the
       original IBO formulation, it is the Hartree-Fock canonical orbitals.
    loc:
       The localization method, available options are 'IBO' and 'PM'

    Outputs
    -------
    iboc:
       The coefficients of IBO-c in AO basis.
    '''

    if mol is None:
        mol = util_atoms.mole(logbook)
        
    assert loc == 'IBO' or loc == 'PM'
    ovl = mol.intor('int1e_ovlp')
    if (oiao is None or ibo is None) and (mo_ref is None):
        mf = scf.RHF(mol).run()
        mo_ref = mf.mo_coeff[:,mf.mo_occ>0]

    #==== Obtain orthogonalized IAOs ====#
    if oiao is None:
        iao = lo.iao.iao(mol, mo_ref)
        oiao = lo.vec_lowdin(iao, ovl)
        
    #==== Obtain IBOs ====#
    if ibo is None:
        ibo = get_IBO(mol, oiao, mo_ref, by_symm, align_groups)

    #==== Obtain OIAOs in symm. orthogonalized basis ====#
    e, v = scipy.linalg.eigh(ovl)
    x = v @ np.diag( 1/np.sqrt(e) ) @ v.T   # X for symmetric orthogonalization
    x_i = v @ np.diag( np.sqrt(e) ) @ v.T   # X^-1 for symmetric orthogonalization
    oiao_ = x_i @ oiao
    ibo_ = x_i @ ibo
    n_iboc = oiao_.shape[1] - ibo_.shape[1]

    #==== Project OIAO into the complementary space of the IBOs ====#
    # If ibo_ has definite symmetry (irrep), then Q_proj conserves the symmetry of the
    # input vector.
    Q_proj = np.eye(ibo_.shape[0]) - ibo_ @ ibo_.T
    iboc = Q_proj @ oiao_
    norms = np.einsum('ij, ji -> i', iboc.T, iboc)
    iboc = iboc / np.sqrt(norms)

    #==== Orthogonalize IBOC via SVD ====#
    if by_symm:
        # As it turns out, orthogonalization through SVD does not necessarily conserve
        # symmetry. This is especially true in linear molecules with point group set to
        # D2h or C2v where the absence of symmetry is reflected in the lobes of pi
        # orbitals not oriented along any Cartesian axes. This block performs SVD for
        # each symmetry block of iboc obtained above, hence preserving the symmetry
        # of these iboc's.
        o = np.array([])
        iboc_s = np.array( symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, x@iboc) )
        n_ortho = 0
        for s in set(iboc_s):
            U, sv, Vt = np.linalg.svd(iboc[:,iboc_s == s], full_matrices=False)
            nsym = len(sv[sv > ortho_thr])
            o = U[:,0:nsym] if o.size == 0 else np.hstack((o, U[:,0:nsym]))
            n_ortho += nsym
        iboc = o[:,0:n_ortho]
    else:
        U, sv, Vt = np.linalg.svd(iboc, full_matrices=False)
        n_ortho = len(sv[sv > ortho_thr])
        iboc = U[:,0:n_ortho]
    
    assert n_ortho == n_iboc, f'The number of retained IBOCs ({n_ortho}) must be ' + \
        'the same as the difference between the number of AOs and the number of ' + \
        f'IBOs ({n_iboc}). Try changing ortho_thr.'    
        
    #==== Express orthogonalized IBOC in AO basis ====#
    iboc = x @ iboc

    #==== Localize IBOC ====#
    if loc == 'IBO':
        #iboc = lo.ibo.ibo(mol, iboc, iaos=oiao)
        iboc = get_IBO(mol, oiao, iboc, by_symm, align_groups)
    elif loc == 'PM':
        assert by_symm, 'When using PM is localization method, by_symm must be True.'
        outs = local_orbs.localize(mol, iboc, loc_subs=[[i+1 for i in range(iboc.shape[1])]])
        iboc = outs['orbs']

    return iboc
##########################################################################


##########################################################################
def analyze(ibo, oiao, mol=None, iboc=None, print_ibo=False, print_oiao=False,
            print_iboc=False, cube_dir=None, logbook=None):

    if mol is None:
        mol = util_atoms.mole(logbook)
    
    ovl = mol.intor('int1e_ovlp')
    print('Number of electrons in mol object = ', mol.nelectron)
    print('Number of AOs = ', mol.nao)
    if iboc is not None:
        print('Sizes of IAO, IBO, and IBOC = ', oiao.shape[1], ibo.shape[1], iboc.shape[1])
    else:
        print('Sizes of IAO and IBO = ', oiao.shape[1], ibo.shape[1])
    print('Trace of overlap matrix of IBO in AO rep. = %10.6f' % np.trace(ibo.T @ ovl @ ibo))
    if iboc is not None:
        print('Trace of overlap matrix of IBOC in AO rep. = %10.6f' %
              np.trace(iboc.T @ ovl @ iboc))
        print('Frobenius norm of overlap matrix between IBOC and IBO = %10.6f' %
              np.linalg.norm(iboc.T @ ovl @ ibo, ord='fro') )

    #==== Analyze IAOs ====#
    print('')
    uprint.print_section('Analysis of IAOs')
    analyze_orbs.analyze(mol, oiao)
    
    #==== Analyze IBOs ====#
    print('')
    uprint.print_section('Analysis of IBOs')
    analyze_orbs.analyze(mol, ibo)
    
    #==== Analyze IBOC's ====#
    if iboc is not None:
        print('')
        uprint.print_section('Analysis of IBOC\'s')
        analyze_orbs.analyze(mol, iboc)

    #==== Print IAOs, IBOs, and IBOCs into cube files ====#
    if print_oiao or print_ibo:
        assert cube_dir is not None, 'cube_dir is needed when printing IBO or IAO into ' + \
            'cube files.'
        ndigit_oiao = len(str(oiao.shape[1]))
        ndigit_ibo = len(str(ibo.shape[1]))
        if not os.path.isdir(cube_dir):
            os.mkdir(cube_dir)
        for i in range(0, max(oiao.shape[1], ibo.shape[1])):
            if print_oiao and i < oiao.shape[1]:
                cubename = cube_dir + '/iao-' + str(i+1).zfill(ndigit_oiao) + '.cube'
                print(f'{i+1:d}) Printing cube files: ', flush=True)
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, oiao[:,i])
            if print_ibo and i < ibo.shape[1]:
                cubename = cube_dir + '/ibo-' + str(i+1).zfill(ndigit_ibo) + '.cube'
                print(f'{i+1:d}) Printing cube files: ', flush=True)
                print('     ' + cubename)
                tools.cubegen.orbital(mol, cubename, ibo[:,i])
    if iboc is not None and print_iboc:
        assert cube_dir is not None, 'cube_dir is needed when printing IBOC into ' + \
            'cube files.'
        ndigit_iboc = len(str(iboc.shape[1]))
        for i in range(0, iboc.shape[1]):
            cubename = cube_dir + '/iboc-' + str(i+1).zfill(ndigit_iboc) + '.cube'
            print(f'{i+1:d}) Printing cube files: ', flush=True)
            print('     ' + cubename)
            tools.cubegen.orbital(mol, cubename, iboc[:,i])
##########################################################################
