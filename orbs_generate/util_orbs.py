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
    idsort = idsort + [i for i in range(0,len(osym)) if osym[i] not in symset]
    idsort = np.array(idsort)
    
    return idsort
##########################################################################


##########################################################################
def ortho_project(mol, a, b, ortho_thr=1E-8, verbose=2):
    '''
    This function projects columns of b onto the orthogonal space of columns of a.
    Then it removes the ensuing linear dependent vectors from among the result of
    the projection and orthogonalize them. It returns the vector resulting from
    these steps.
    a = Orbitals in AO basis.
    b = Orbitals in AO basis.
    '''

    
    ovl = mol.intor('int1e_ovlp')   # S
    e, v = scipy.linalg.eigh(ovl)
    iovl = v @ np.diag(1/e) @ v.T   # S^(-1)
    ovl12 = v @ np.diag(np.sqrt(e)) @ v.T   # S^(1/2)
    iovl12 = v @ np.diag(1/np.sqrt(e)) @ v.T   # S^(-1/2)

    nb = b.shape[1]

    #==== Project onatc onto the ortho. space of obase ====#
    Q = ovl - ovl @ a @ a.T @ ovl
    cc0 = iovl @ Q @ b

    #==== Remove zero columns in cc0 ====#
    # Some columns of b might be in the span of columns of a, the projection of
    # these columns by Q produces zero vectors.
    nn = np.sqrt( np.einsum('ij,jk,ki -> i', cc0.T, ovl, cc0) )
    cc = cc0[:,nn > 1E-8]
    
    #==== Perform orthogonolization and remove linear dependence ====#
    ortho = 'svd'     # 1)
    # 1) ortho = 'ovl' is not recommended at the moment because the correct way 
    #    of removing linear dependent vectors is not yet implemented.
    if ortho == 'svd':
        # Here, the SVD is performed directly within the non-orthogonal AO basis.
        # To do this, a SVD decomposition is peformed on S^(1/2).C,
        #    S^(1/2).C = U.s.V^T
        # and we define U' such that U = S^(1/2).U'. Here
        #    C := cc
        #    S := ovl
        # The orthogonalized vectors are then the columns of U'. It can be checked
        # that the orthogonality of U induces the orthogonality of U' under metric
        # S, i.e.
        #    U^T.U = I --> U'^T.S.U' = I
        # and that the span of U' is the same as the span of C since
        #    C = U'.s.V^T
        # and s.V^T is right-invertible after removing zero singular values and the
        # corresponding rows of V^T.
        o = np.array([])
        osym = np.array( symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, cc) )
        n_ortho = 0
        for s in set(osym):
            U, sv, Vt = np.linalg.svd(ovl12 @ cc[:,osym == s], full_matrices=False)
            nsym = len(sv[sv > ortho_thr])
            o = iovl12 @ U[:,0:nsym] if o.size == 0 \
                else np.hstack((o, iovl12 @ U[:,0:nsym]))
            n_ortho += nsym
        c = o[:,0:n_ortho]
    elif ortho == 'ovl':
        so = cc.T @ ovl @ cc
        e, v = scipy.linalg.eigh(so)
        v = v[:, np.abs(e) > ortho_thr]
        e = e[np.abs(e) > ortho_thr]
        n_ortho = len(e)
        so12 = v @ np.diag( 1/np.sqrt(e) ) @ v.T
        c = cc @ so12

    if verbose >= 2:
        print('No. of retained vectors = ', n_ortho, ' out of ', nb, ' vectors.')

    return c
##########################################################################

