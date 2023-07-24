import numpy as np
from pyscf import gto, lo, symm


##########################################################################
def localize(orbs, mol, loc_type='PM', occs=None, large_occ=1.9, loc_irrep=True):
    '''
    large_occ = only used when occs is not None.
    orb_sym = onlyused when loc_irrep is True.
    '''

    #==== Obtain the irrep of each orbital ====#
    if loc_irrep:
        orb_sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orbs)
                
    #==== Divide orbs into large and small occupations ====#
    if occs is not None:
        #== occs must be monotonically decreasing ==#
        t = [occs[i] >= occs[i+1] for i in range(0,len(occs)-1)]
        assert all(t), 'localize_orbs: The \'occs\' argument is not monotonically ' + \
            'decreasing. occs = ' + str(occs)

        #== Assign the orbitals with large and small occupations ==#
        n_large_occ = sum(occs > large_occ)
        orbs_ = [None] * 2
        orbs_[0] = orbs[:,:n_large_occ]    # Orbitals with large occupations.
        orbs_[1] = orbs[:,n_large_occ:]    # Orbitals with small occupations.

        if loc_irrep:
            orb_sym_ = [None] * 2
            orb_sym_[0] = orb_sym[:n_large_occ]    # Irrep of orbitals with large occupations.
            orb_sym_[1] = orb_sym[n_large_occ:]    # Irrep of orbitals with small occupations.
    else:
        #== No division ==#
        orbs_ = [None]
        orbs_[0] = orbs

        if loc_irrep:
            orb_sym_ = [None]
            orb_sym_[0] = orb_sym

        
    #==== Perform localization within each irrep ====#
    if loc_type == 'ER':
        do_loc = lo.ER
    elif loc_type == 'PM':
        do_loc = lo.PM
    else:
        raise ValueError('localize: The value of the argument loc_type is undefined, ' + \
                         'loc_type = ' + str(loc_type))
    
    orbs_l = []
    if loc_irrep:
        #== Loop over the large/small occupation sections ==#
        for i in range(0,len(orb_sym_)):
            n = len(orb_sym_[i])
            symset = set(orb_sym_[i])     # Unique elements of the irreps in orb_sym_[i]

            #== Loop over the unique irreps ==#
            for s in symset:
                ids = [k for k in range(0,n) if orb_sym_[i][k]==s]
                orbs_s = orbs_[i][:, ids]
                orbs_l.append( do_loc(mol, orbs_s).kernel() )
    else:
        for i in range(0, len(orbs_)):
            orbs_l.append( do_loc(mol, orbs_[i]).kernel() )

            
    orbs_l = np.hstack(orbs_l)
    return orbs_l
##########################################################################
