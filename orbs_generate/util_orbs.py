import numpy as np
import scipy.linalg
from pyscf import scf, symm, lo
from IMAM_TDDMRG.utils.util_print import print_matrix



##########################################################################
def sort_orbs(orbs, occs, ergs, sr='erg', s='de'):
    '''
    Sort occs, orbs, and ergs based on the elements of occs or ergs.
    '''

    assert isinstance(occs, np.ndarray) or occs is None
    assert isinstance(ergs, np.ndarray) or ergs is None

    
    assert sr == 'erg' or sr == 'occ', \
        'sort_orbs: The value of the argument \'sr\' can only either ' + \
        'be \'erg\' or \'occ\'. Its current value: sr = ' + sr + '.'
    assert s == 'de' or s == 'as', \
        'sort_orbs: The value of the argument \'s\' can only either ' + \
        'be \'de\' or \'as\'. Its current value: s = ' + s + '.'

    if sr == 'erg' and ergs is None:
        raise ValueError('sort_orbs: If sr = \'erg\', then ergs must ' + \
                         'not be None.')
    if sr == 'occ' and occs is None:
        raise ValueError('sort_orbs: If sr = \'occ\', then occs must ' + \
                         'not be None.')

    if s == 'de':       # Descending
        if sr == 'erg': isort = np.argsort(-ergs)
        if sr == 'occ': isort = np.argsort(-occs)
    elif s == 'as':     # Ascending
        if sr == 'erg': isort = np.argsort(ergs)
        if sr == 'occ': isort = np.argsort(occs)
    
    if occs is not None: occs = occs[isort]
    if ergs is not None: ergs = ergs[isort]
    orbs = orbs[:,isort]

    return orbs, occs, ergs
##########################################################################


##########################################################################
def sort_similar(orb, orb_ref, ovl=None, similar_thr=0.8, dissim_break=False, verbose=1):
    '''
    orb: Orbitals to be sorted in AO basis.
    orb_ref: Orbitals in AO basis whose ordering is used as reference for the sorting.
    '''
    
    assert orb.shape[0] == orb_ref.shape[0]
    assert orb.shape[1] == orb_ref.shape[1]
    
    n = orb.shape[0]
    if ovl is None:
        ovl = np.eye(n,n)
        
    ovl_ref = orb_ref.T @ ovl @ orb
    if verbose == 1:
        print('Overlap matrix between the calculated (column) and reference (row) orbitals:')
        print_matrix(ovl_ref)
        
    idmax = [np.argmax(np.abs(ovl_ref[:,i])) for i in range(0, ovl_ref.shape[1])]
    idmax = np.array(idmax)
    #OLDassert len(idmax) == len(set(idmax)), 'Some orbitals have conflicting orders. ' + \
    #OLD    f'ID of the maximum of each column: {idmax}'
    elmax = [np.max(np.abs(ovl_ref[:,i])) for i in range(0, ovl_ref.shape[1])]
    elmax = np.array(elmax)
    dissim = idmax[elmax < similar_thr]
    if len(dissim) > 0:
        if dissim_break:
            raise ValueError('These calculated orbitals: {str(dissim+1)} are ' + \
                             'too dissimilar to the reference orbitals.')
        else:
            print(f'WARNING: These calculated orbitals: {str(dissim+1)} are too ' + \
                  'dissimilar to the reference orbitals.')
    
    # Reorder columns of orb such that if ovl_ref is recalculated using the newly
    # ordered orb, the maximum of each column of ovl_ref is a diagonal element.
    idsort = np.argsort(idmax)

    return idsort
##########################################################################


##########################################################################
##########################################################################
def sort_irrep(mol, orb, irrep=None):

    osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, orb)
    symset = set(osym)
    if irrep is not None:
        assert len(irrep) == len(set(irrep))
        for i in range(0,len(irrep)):
            assert irrep[i] in symset
        symset = irrep
            
    idsort = []
    for s in symset:
        idsort = idsort + [i for i in range(0,len(osym)) if osym[i]==s]
    idsort = np.array(idsort)
    
    return idsort
##########################################################################


##########################################################################
def get_IBO(mol, mf=None, align_groups=None):
    
    if align_groups is not None:
        assert isinstance(align_groups, (list, tuple)), 'align_groups must be a tuple or list ' + \
            'which in turn contains lists or tuples.'

    ovl = mol.intor('int1e_ovlp')
    if mf is None: mf = scf.RHF(mol).run()
    
    #==== Obtain IAOs in AO basis ====#
    mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
    iao_ = lo.iao.iao(mol, mo_occ)
    
    #==== Obtain IAOs in symm. orthogonalized basis ====#
    e, v = scipy.linalg.eigh(ovl)
    x_i = v @ np.diag( np.sqrt(e) ) @ v.T   # X^-1 for symmetric orthogonalization
    iao = x_i @ iao_
    
    #==== Calculate IBOs in AO basis ====#
    ibo_ = lo.ibo.ibo(mol, mo_occ, iaos=iao)
    if align_groups is not None:
        # The task in this block is to determine mix_coef that satisfies:
        #     I = mo_ref.T @ ovl @ ibo_new
        # where
        #     ibo_new = ibo_old @ mix_coef
        # ibo_new is needed in linear molecules because the lobes of ibo_old are not necessarily aligned along
        # any Cartesian axes. The first equation holds under the assumption that the irrep orderings in mo_ref and
        # ibo_new are the same. Hence, the irrep ordering of ibo_new follows that of mo_ref. The columns of ibo_old
        # must be degenerate, likewise, the columns of mo_ref may not be degenerate but better are.
        mo_irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mf.mo_coeff[:,0:ibo_.shape[1]])
        for ik in align_groups:
            ip = tuple( [ik[j]-1 for j in range(0,len(ik))] )
            print('  -> Aligning IBOs:', ip, '(0-based)')
    
            #== Find the most suitable MOs for mo_ref ==#
            c = mf.mo_coeff.T @ ovl @ ibo_[:,ip]
            idx = np.argmax(np.abs(c), axis=0)
            print('       Indices of the most suitable reference MOs:', idx, '(0-based)')
            mo_ref = mf.mo_coeff[:, idx]
            ref_irreps = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_ref)
    
            #== Compute the new (aligned) IBOs ==#
            mix_coef = np.linalg.inv(mo_ref.T @ ovl @ ibo_[:,ip])
            ibo0 = ibo_[:,ip] @ mix_coef
            norms = np.einsum('ij, jk, ki -> i', ibo0.T, ovl, ibo0)  # np.diag(ibo_.T @ ovl @ ibo_)
            ibo_[:,ip] = ibo0 / np.sqrt(norms)

    return ibo_, iao_
##########################################################################


##########################################################################
def get_IBOC(mol, boao=None, mf=None, loc='IBO', align_groups=None):
    '''
    This function calculates the set of vectors orthogonal to IBOs (calculated by get_IBO) that
    are also spanned by IAOs.

    Inputs
    ------
    boao:
       A tuple of the form (IBO, IAO) in AO basis.
    loc:
       The localization method, available options are 'IBO' and 'PM'

    Outputs
    -------
    ibo_c:
       The coefficients of IBO-c in AO basis.
    '''
     
    assert loc == 'IBO' or loc == 'PM'
    
    if boao is not None:
        assert isinstance(boao, tuple)
        ibo_, iao_ = boao
    else:
        ibo_, iao_ = get_IBO(mol, mf, align_groups)
    ovl = mol.intor('int1e_ovlp')

    #==== Obtain IAOs in symm. orthogonalized basis ====#
    e, v = scipy.linalg.eigh(ovl)
    x = v @ np.diag( 1/np.sqrt(e) ) @ v.T   # X for symmetric orthogonalization
    x_i = v @ np.diag( np.sqrt(e) ) @ v.T   # X^-1 for symmetric orthogonalization
    iao = x_i @ iao_
    ibo = x_i @ ibo_
    
    #==== Select the most suitable IAOs to construct IBO-c ====#
    # Here, the first n_ibo_c columns of IAO with the smallest IBO components are
    # selected as the vectors from which IBO-c is constrcuted. The idea of this
    # choice is that IBO-c is supposed to be orthogonal to IBO, hence it makes sense
    # to construct the former by assuming small contribution from the latter.
    n_ibo_c = iao.shape[1] - ibo.shape[1]
    norms = np.linalg.norm(ibo.T @ iao, axis=0)
    ibo_c_id = np.argsort(norms)[0:n_ibo_c]

    #==== Project IAO into the complementary space of the IBOs ====#
    Q_proj = np.eye(ibo.shape[0]) - ibo @ ibo.T
    ibo_c = Q_proj @ iao[:,ibo_c_id]
    norms = np.einsum('ij, ji -> i', ibo_c.T, ibo_c)
    ibo_c = ibo_c / np.sqrt(norms)

    #==== Orthogonalize IBO-c in Lowdin basis ====#
    ibo_c, R = scipy.linalg.qr(ibo_c, mode='economic')

    #==== Express orthogonalized IBO-c in AO basis ====#
    ibo_c = x @ ibo_c

    #==== Localize IBO-c ====#
    if loc == 'IBO':
        ibo_c = lo.ibo.ibo(mol, ibo_c, iaos=iao)
    elif loc == 'PM':
        ibo_c = lo.PM(mol, ibo_c).kernel()

    return ibo_c
##########################################################################
