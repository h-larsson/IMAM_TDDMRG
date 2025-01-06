import numpy as np
from pyscf import gto, lo, symm
from TDDMRG_CM.orbs_generate.util_orbs import sort_orbs


loc_type_err = 'localize: The value of the argument \'loc_type\' is undefined, ' + \
               'The available options are \'ER\', \'PM\', \'B\', and a numpy ' + \
               'array.'

##########################################################################
def localize(mol, orbs0, rdm0=None, ovl=None, loc_subs=None, occs_thr=None, loc_type=None,
             loc_irrep=None, excludes=[], sort_to_occ=False):
    '''
    mol:
      The Mole object based on which the input orbitals is constructed.
    orbs0:
      Input orbitals to be localized.
    rdm0:
      The 1RDM in the input orbital basis, hence, e.g., if the input orbitals are 
      the natural orbitals of some state, then rdm0 should be a diagonal matrix.
    loc_subs:
      A list of lists where each list element contains the 1-base orbital indices
      to be localized in the localization subspace represented by this list element.
      That is, the lists in loc_subs represent the localization subspaces.
    occs_thr:
      The occupation thresholds that define the localization subspaces.
      If there are more than two subspaces, it must be a list with
      monotonically increasing elements ranging between but does not
      include 0.0 and 2.0.
    loc_type:
      Orbital localization type, the supported values are 'PM' (Pipek-Mezey, the
      default), 'ER' (Edmiston-Ruedenberg), and 'B' (Boys).
    loc_irrep:
      If True, then the orbitals will be localized within each irreducible 
      representation of the point group of the molecule. Useful for preventing 
      symmetry breaking of orbitals as a result of the localization. Therefore,
      unless absolutely needed, this argument should always be left to its default.
      value.
    sort_to_occ:
      Only meaningful when rdm0 is not None. If True, the output localized orbitals
      will be sorted according to their occupation numbers calculated using rdm0.
    '''

    print('')
    assert (loc_subs is not None) ^ (occs_thr is not None), \
        'The existence of the inputs loc_subs and occs_thr is mutually exclusive. If one of ' + \
        'them is present, the other one must be absent.'
    has_rdm0 = (rdm0 is not None)

    
    #== Determine orbital IDs in each subspace based on occupations ==#
    if occs_thr is not None:
        assert has_rdm0
        occs0 = np.diag(rdm0).copy()          # occs0 = Occupations of input orbitals.
        occs_id = np.array( [ i+1 for i in range(0,len(occs0)) ] )   # i+1 instead of i because it has to be 1-base.
        occs_thr = [-0.1] + occs_thr + [2.1]           # occs_thr = Occupation thresholds to determine localization subspaces.
        # The last element of occs_thr above is purposely set to 2.1 instead of 2.0 because
        # the double occupations elements of occs0 are sometimes very slightly over 2.0.
        # The same reason for the first element of occs_thr.
        
        loc_subs_occs = [None] * (len(occs_thr)-1)    # loc_subs_occs = Orbital indices in each localization subspace.
        for i in range(0, len(occs_thr)-1):
            if i < len(occs_thr)-2:
                cond = np.array( (occs0 >= occs_thr[i]) & (occs0 < occs_thr[i+1]) )
            else:
                cond = np.array( (occs0 >= occs_thr[i]) & (occs0 <= occs_thr[i+1]) )
            loc_subs_occs[i] = list(occs_id[cond])
        nsubs = len(loc_subs_occs)
    else:
        nsubs = len(loc_subs)

            
    #== Do localization ==#
    if loc_type == None:
        loc_type = ['PM'] * nsubs
    if loc_irrep == None:
        loc_irrep = [True] * nsubs
    orbs = orbs0.copy()
    for i in range(0, nsubs):
        # iloc0 is the 1-based indices of the current localization space.
        # iloc is the 0-based indices of the current localization space.
        if occs_thr is not None:
            iloc0 = loc_subs_occs[i]
        else:
            iloc0 = loc_subs[i]

        #== Remove excluded orbitals from the current subspace ==#
        iloc0 = [e for e in iloc0 if e not in excludes]
        iloc = [iloc0[j]-1 for j in range(0,len(iloc0))]

        #== Localize orbitals whose IDs are in iloc ==#
        print('>>> Performing localization <<<')
        print('  Localization method = ', loc_type[i])
        print('  IDs of orbitals to be localized = ', iloc0)
        orbs[:,iloc] = localize_sub(orbs0[:,iloc], mol, loc_type[i], 
                       loc_irrep[i])

        
    #== Transformation matrix between initial and localized orbitals ==#
    if ovl is None: ovl = mol.intor('int1e_ovlp')
    trmat = orbs0.T @ ovl @ orbs

    
    #== Sort orbitals and occupations based on occupations if initial RDM is available ==#
    if has_rdm0:
        occs = np.zeros(orbs.shape[1])
        for i in range(0, orbs.shape[1]):
            occs[i] = np.einsum('j, jk, k', trmat[:,i], rdm0, trmat[:,i])
        if sort_to_occ:
            orbs, occs, _ = sort_orbs(orbs, occs, None, 'occ', 'de')
            #== Reorder trmat ==#
            trmat = orbs0.T @ ovl @ orbs
        rdm = trmat.T @ rdm0 @ trmat    # rdm is the 1RDM in the localized orbitals basis.
    else:
        occs = None
        rdm = None
        

    #== What to output ==#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    outs['coef'] = trmat
    if has_rdm0: outs['rdm'] = rdm

    print('')
    return outs
##########################################################################


##########################################################################
def localize_sub(orbs, mol, loc_type='PM', loc_irrep=True):
    '''
    orb_sym = onlyused when loc_irrep is True.
    '''

    #==== Obtain the irrep of each orbital ====#
    if loc_irrep:
        orb_sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orbs)
        
        
    #==== Perform localization within each irrep ====#
    if isinstance(loc_type, str):
        if loc_type == 'ER':
            do_loc = lo.ER
        elif loc_type == 'PM':
            do_loc = lo.PM
        elif loc_type == 'B':
            do_loc = lo.Boys
        else:
            raise ValueError(loc_type_err)
    elif isinstance(loc_type, np.ndarray):
        do_loc = loc_type.copy()
        assert do_loc.shape[0] == do_loc.shape[1]     # loc_type must be a square matrix.
        itest = do_loc.T @ do_loc
        assert abs(np.sum(itest) - do_loc.shape[0]) < 1.0E-12       # Columns of loc_type must be orthonormal.
    else:
        raise ValueError(loc_type_err)

    
    symset = set(orb_sym)     # Unique elements of the irreps in orb_sym[i]
    print('  Detected irreps among the input orbitals = ', symset)

    orbs_l = np.zeros(orbs.shape)
    if loc_irrep:
        #== Loop over the large/small occupation sections ==#
        n = len(orb_sym)

        #== Loop over the unique irreps ==#
        for s in symset:
            ids = [k for k in range(0,n) if orb_sym[k]==s]
            if loc_type == 'ER' or loc_type == 'PM' or loc_type == 'B':
                orbs_l[:,ids] = do_loc(mol, orbs[:,ids]).kernel()
            elif isinstance(loc_type, np.ndarray):
                orbs_l[:,ids] = orbs[:,ids] @ do_loc
    else:
        print('  Irreps are disregarded in the localization.')
        if loc_type == 'ER' or loc_type == 'PM' or loc_type == 'B':
            orbs_l = do_loc(mol, orbs).kernel()
        elif isinstance(loc_type, np.ndarray):
            orbs_l = orbs @ do_loc
                            
    return orbs_l
##########################################################################
