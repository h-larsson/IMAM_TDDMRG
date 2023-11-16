import numpy as np
from pyscf import gto, lo, symm


loc_type_err = 'localize: The value of the argument \'loc_type\' is undefined, ' + \
               'The available options are \'ER\', \'PM\', \'B\', and a numpy ' + \
               'array.'

##########################################################################
def localize(orbs, mol, loc_type='PM', loc_irrep=True):
    '''
    loc_type = Orbital localization type, the supported values are 'PM' (Pipek-Mezey, the
               default), 'ER' (Edmiston-Ruedenberg), and 'B' (Boys). Only meaningful when 
               loc_orb = True.
    loc_irrep = If True, then the orbitals will be localized within each irreducible 
                representation of the point group of the molecule. Useful for preventing 
                symmetry breaking of the orbitals as a result of the localization. 
                Therefore, unless absolutely needed, this argument should always be True. 
                Only meaningful when loc_orb = True. 
    '''
    
    print('>>> Performing localization <<<')
    if isinstance(loc_type, str):
        if loc_type == 'PM': loc_type_ = 'Pipek-Mezey'
        elif loc_type == 'ER': loc_type_ = 'Edmiston-Ruedenberg'
        elif loc_type == 'B': loc_type_ = 'Boys'
        else: raise ValueError(loc_type_err)
    elif isinstance(loc_type, np.ndarray):
        loc_type_ = 'Manual'
    else:
        raise ValueError(loc_type_err)
    print('Localization type = %s' % loc_type_)
        
    orbs_ = localize_sub(orbs, mol, loc_type, loc_irrep)

    return orbs_

##########################################################################


##########################################################################
def localize_sub(orbs, mol, loc_type='PM', loc_irrep=True):
    '''
    orb_sym = onlyused when loc_irrep is True.
    '''

    #==== Obtain the irrep of each orbital ====#
    if loc_irrep:
        orb_sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, orbs)
                
    #==== Divide orbs into large and small occupations ====#
    #== No division ==#
    orbs_ = orbs
    if loc_irrep:
        orb_sym_ = orb_sym
        
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
    
    orbs_l = []
    if loc_irrep:
        #== Loop over the large/small occupation sections ==#
        n = len(orb_sym_)
        symset = set(orb_sym_)     # Unique elements of the irreps in orb_sym_[i]

        #== Loop over the unique irreps ==#
        for s in symset:
            ids = [k for k in range(0,n) if orb_sym_[k]==s]
            orbs_s = orbs_[:, ids]
            if loc_type == 'ER' or loc_type == 'PM' or loc_type == 'B':
                orbs_l.append( do_loc(mol, orbs_s).kernel() )
            elif isinstance(loc_type, np.ndarray):
                orbs_l.append( orbs_s @ do_loc )
    else:
        if loc_type == 'ER' or loc_type == 'PM' or loc_type == 'B':
            orbs_l.append( do_loc(mol, orbs_).kernel() )
        elif isinstance(loc_type, np.ndarray):
            orbs_l.append( orbs_ @ do_loc )
                
            
    orbs_l = np.hstack(orbs_l)
    return orbs_l
##########################################################################
