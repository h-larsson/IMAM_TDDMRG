import numpy as np
from pyscf import gto, lo, symm


##########################################################################
def localize(orbs, mol, loc_type='PM', loc_irrep=True, rdm_mo=None):
    '''
    The occupations of the output localized orbitals (occs_) will be returned only if
    rdm_mo is not None.
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
    if loc_type == 'PM': loc_type_ = 'Pipek-Mezey'
    if loc_type == 'ER': loc_type_ = 'Edmiston-Ruedenberg'
    if loc_type == 'B': loc_type_ = 'Boys'        
    print('Localization type = %s' % loc_type_)

    if rdm_mo is not None:
        ovl = mol.intor('int1e_ovlp') 
        rdm_ao = reduce(np.dot, (ovl, orbs, rdm_mo, orbs.T, ovl))    # rdm_ao is in AO rep.
        
    orbs_ = localize_sub(orbs, mol, loc_type, loc_irrep)

    #==== Calculate occupations of the loc. orbitals if rdm_mo is not None ====#
    if rdm_mo is not None:
        occs_ = [None] * orbs.shape[1]
        for i in range(0, orbs.shape[1]):
            occs_[i] = np.einsum('j, jk, k', orbs[:,i], rdm_ao, orbs[:,i])

        orbs_, occs_, ergs_ = sort_orbs(orbs_, occs_, None, 'occ', 'de')
        return orbs_, occs_
    else:
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
    if loc_type == 'ER':
        do_loc = lo.ER
    elif loc_type == 'PM':
        do_loc = lo.PM
    elif loc_type == 'B':
        do_loc = lo.Boys
    else:
        raise ValueError('localize: The value of the argument \'loc_type\' is undefined, ' + \
                         'loc_type = ' + str(loc_type) + '. The available options are ' + \
                         '\'ER\', \'PM\', and \'B\'.')
    orbs_l = []
    if loc_irrep:
        #== Loop over the large/small occupation sections ==#
        n = len(orb_sym_)
        symset = set(orb_sym_)     # Unique elements of the irreps in orb_sym_[i]

        #== Loop over the unique irreps ==#
        for s in symset:
            ids = [k for k in range(0,n) if orb_sym_[k]==s]
            orbs_s = orbs_[:, ids]
            orbs_l.append( do_loc(mol, orbs_s).kernel() )
    else:
        orbs_l.append( do_loc(mol, orbs_).kernel() )

            
    orbs_l = np.hstack(orbs_l)
    return orbs_l
##########################################################################
