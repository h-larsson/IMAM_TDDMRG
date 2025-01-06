import numpy as np
from pyscf import symm
from block2 import SU2, SZ
from gfdmrg import orbital_reorder
from TDDMRG_CM.utils.util_print import _print

# Set spin-adapted or non-spin-adapted here
SpinLabel = SU2
#SpinLabel = SZ




#################################################
def make_full_dm(ncore, dm):
    assert len(dm.shape) == 3
    assert dm.shape[0] == 2
    assert dm.shape[1] == dm.shape[2]
    
    complex_dm = (type(dm[0,0,0]) == np.complex128)
    dtype = np.complex128 if complex_dm else np.float64
    nact = dm.shape[2]
    nocc = ncore + nact
    
    dm_ = np.zeros((2, nocc, nocc), dtype=dtype)
    for i in range(0,2):
        if complex_dm:
            dm_[i, 0:ncore, 0:ncore] = np.diag( [complex(1.0, 0.0)]*ncore )
        else:
            dm_[i, 0:ncore, 0:ncore] = np.diag( [1.0]*ncore )
        dm_[i, ncore:nocc, ncore:nocc] = dm[i, :, :]

    return dm_
#################################################


#################################################
def get_CAS_ints(mol, nCore, nCAS, nelCAS, ocoeff, verbose):
    '''
    Input parameters:
       mol     : PYSCF Mole object that defines the system of interest.
       nCore   : The number of core orbitals.
       nCAS    : The number of CAS orbitals.
       nelCAS  : The number of electrons in the CAS.
       ocoeff  : Coefficients of all orbitals (core+CAS+virtual) in the AO basis 
                 used in mol (Mole) object.
       verbose : Verbose output when True.

    Return parameters:
       h1e         : The one-electron integral matrix in the CAS orbital basis, the size 
                     is nCAS x nCAS.
       g2e         : The two-electron integral array in the CAS orbital basis.
       eCore       : The core energy, it contains the core electron energy and nuclear 
                     repulsion energy.
       molpro_oSym : The symmetry ID of the CAS orbitals in MOLPRO convention.
       molpro_wSym : The symmetry ID of the wavefunction in MOLPRO convention.
    '''

    from pyscf import scf, mcscf, symm
    from pyscf import tools as pyscf_tools
    from pyscf.mcscf import casci_symm


    if SpinLabel == SZ:
        _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        _print('WARNING: SZ Spin label is chosen! The get_CAS_ints function was ' +
               'designed with the SU2 spin label in mind. The use of SZ spin label in ' +
               'this function has not been checked.')
        _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    #forlater irname = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, ocoeff)
    #forlater print('here ocoeff irname = ', irname)
    
    
    #==== Setting up the CAS ====#
    mf = scf.RHF(mol)
    _mcCI = mcscf.CASCI(mf, ncas=nCAS, nelecas=nelCAS , ncore=nCore)  # IMAM: All orbitals are used?
    _mcCI.mo_coeff = ocoeff          # IMAM : I am not sure if it is necessary.
    _mcCI.mo_coeff = casci_symm.label_symmetry_(_mcCI, ocoeff)
    

    #==== Get the 1e and 2e integrals ====#
    h1e, eCore = _mcCI.get_h1cas()
    g2e = _mcCI.get_h2cas()
    g2e = np.require(g2e, dtype=np.float64)
    h1e = np.require(h1e, dtype=np.float64)

    del _mcCI, mf

    return h1e, g2e, eCore
#################################################


#################################################
def get_syms(mol, wsym, nCore, nCAS, ocoeff):
    '''
    Input parameters:
       wsym   : The symmetry name or ID (in pyscf convention) of the total wavefunction.
       ocoeff : Coefficients of all orbitals (core+CAS+virtual) in the AO basis used in mol
                (Mole) object.
    Return parameters:
       molpro_oSym : The symmetry ID of the CAS orbitals in MOLPRO convention.
       molpro_wSym : The symmetry ID of the total wavefunction in MOLPRO convention.
    '''
    from pyscf import symm
    from pyscf import tools as pyscf_tools
    

    #==== Wavefunction symmetry ====#
    if isinstance(wsym, str):
        wsym_ = symm.irrep_name2id(mol.groupname, wsym)
    elif isinstance(wsym, int):
        wsym_ = wsym
    else:
        raise ValueError('The argument wsym of get_syms must be either an integer or a string.')
    molpro_wsym = pyscf_tools.fcidump.ORBSYM_MAP[mol.groupname][wsym_]

    
    #==== Orbitals symmetry ====#
    osym_l = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, ocoeff))
    osym = [symm.irrep_name2id(mol.groupname, s) for s in osym_l]
    osym = np.array(osym)[nCore:nCore+nCAS]
    molpro_osym = [pyscf_tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in osym]
    
    
    return molpro_osym, molpro_wsym
#################################################


#################################################
def get_one_pdm(iscomp, spin_symm, n_sites, hamil, mps=None, mps_path=None, dmargin=0, 
                ridx=None, mpi=None, verbose=2):

    import time
    import block2
    
    if iscomp:
        bx = block2.cpx
        bc = bx
    else:
        bx = block2
        bc = None    #OLD block2.cpx if has_cpx else None

    if spin_symm == 'su2':
        bs = bx.su2
        brs = block2.su2
        SX = block2.SU2
    elif spin_symm == 'sz':
        bs = bx.sz
        brs = block2.sz
        SX = block2.SZ

    #from bs import SimplifiedMPO, RuleQC, PDM1MPOQC, Expect, ComplexExpect, MovingEnvironment
    #if mpi is not None:
    #    from bs import ParallelMPO
    #from brs import MPSInfo

            
    if mps is None and mps_path is None:
        raise ValueError("The 'mps' and 'mps_path' parameters of "
                         + "get_one_pdm cannot be both None.")
    
    if verbose >= 2:
        _print('>>> START one-pdm <<<')
    t = time.perf_counter()

    if mpi is not None:
        mpi.barrier()

    if mps is None:   # mps takes priority over mps_path, the latter will only be used if the former is None.
        mps_info = brs.MPSInfo(0)
        mps_info.load_data(mps_path)
        mps = MPS(mps_info)
        mps.load_data()
        mps.info.load_mutable()
        
    max_bdim = max([x.n_states_total for x in mps.info.left_dims])
    if mps.info.bond_dim < max_bdim:
        mps.info.bond_dim = max_bdim
    max_bdim = max([x.n_states_total for x in mps.info.right_dims])
    if mps.info.bond_dim < max_bdim:
        mps.info.bond_dim = max_bdim

                    
    # 1PDM MPO
    print('mpi = ', mpi)
    pmpo = bs.PDM1MPOQC(hamil)
    pmpo = bs.SimplifiedMPO(pmpo, bs.RuleQC())
    if mpi is not None:
        if pdmrule is None:
            pdmrule_ = ParallelRuleNPDMQC(mpi)
            pmpo = bs.ParallelMPO(pmpo, pdmrule_)
        else:
            pmpo = bs.ParallelMPO(pmpo, pdmrule)

        
    # 1PDM
    _print(type(pmpo), type(mps))
    pme = bs.MovingEnvironment(pmpo, mps, mps, "1PDM")
    pme.init_environments(False)
    if iscomp:
        expect = bs.ComplexExpect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
    else:
        expect = bs.Expect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
    expect.iprint = max(verbose - 1, 0)
    expect.solve(True, mps.center == 0)
    if spin_symm == 'su2':
        dmr = expect.get_1pdm_spatial(n_sites)
        dm = np.array(dmr).copy()
    elif spin_symm == 'sz':
        dmr = expect.get_1pdm(n_sites)
        dm = np.array(dmr).copy()
        dm = dm.reshape((n_sites, 2, n_sites, 2))
        dm = np.transpose(dm, (0, 2, 1, 3))

    if ridx is not None:
        dm[:, :] = dm[ridx, :][:, ridx]

    mps.save_data()
    if mps is None:
        mps_info.deallocate()
    dmr.deallocate()
    pmpo.deallocate()

    if verbose >= 2:
        _print('>>> COMPLETE one-pdm | Time = %.2f <<<' %
               (time.perf_counter() - t))

    if spin_symm == 'su2':
        return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
    elif spin_symm == 'sz':
        return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
#################################################


#################################################
def orbital_reorder_mrci(ord_method, mol, nCAS1, nCAS2, nelCAS, nelCAS2, orbs):
    '''
    orbs = active orbitals where the first nCAS1 columns are the active1 orbitals and 
           next nCAS2 columns are the active2 orbitals.
    nelCAS = The number of electrons in the total active space.
    nelCAS2 = The number of electrons in the second active space, i.e. the maximum
              excitation order.
    '''
    
    #== 1st acitve space ==#
    h1e_ac1, g2e_ac1, _ = \
        get_CAS_ints(mol, 0, nCAS1, nelCAS, orbs[:, 0:nCAS1], True)
    rid_ac1 = orbital_reorder(h1e_ac1, g2e_ac1, ord_method)

    #== 2nd active space ==#
    h1e_ac2, g2e_ac2, _ = \
        get_CAS_ints(mol, 0, nCAS2, nelCAS2, orbs[:, nCAS1:nCAS1+nCAS2], True)
    rid_ac2 = orbital_reorder(h1e_ac2, g2e_ac2, ord_method) + nCAS1

    return np.hstack((rid_ac1, rid_ac2))
#################################################


#################################################
def orbital_reorder_dip(mol, orbs, uv, verb=4):
    '''
    uv = The vector that defines the direction of component of the dipole 
         moment used for the sorting. It does not have to be of unit length
         because normalization will be done by this function.
    '''
    
    assert len(orbs.shape) == 2, 'The input orbitals to the orbital_reorder_dip ' + \
           'function must be a two-dimensional array (a matrix).'
    assert orbs.shape[0] == mol.nao
    norbs = orbs.shape[1]
    uv = np.array(uv)
    uv = uv / np.sqrt(np.dot(uv,uv))
    
    #==== The multipole operator matrices in AO rep. ====#
    dip_ao = mol.intor('int1e_r').reshape(3,mol.nao,mol.nao)

    #==== Dipole lengths ====#
    dip_l = np.zeros(norbs)
    if verb > 2: _print('Dipole moment of orbitals:')
    for i in range(0, norbs):
        dip = np.einsum('j, xjk, k -> x', orbs[:,i], dip_ao, orbs[:,i])
        dip_l[i] = np.dot(uv, dip)
        if verb >= 2:
            _print('%d' % (i+1), end='')
            for j in range(0,3):
                _print('%11.6f' % dip[j], end='')
            _print('%11.6f' % dip_l[i])

    #==== Ordering ====#
    order_id = np.argsort(dip_l)
        
    return order_id
#################################################


#################################################
def orbital_reorder_mrci_dip(mol, orbs, uv, nCAS1, nCAS2, verb=4):
    '''
    orbs = active orbitals where the first nCAS1 columns are the active1 orbitals and 
           next nCAS2 columns are the active2 orbitals.
    '''
    
    rid_ac1 = orbital_reorder_dip(mol, orbs[:, 0:nCAS1], uv, verb)
    rid_ac2 = orbital_reorder_dip(mol, orbs[:, nCAS1:nCAS1+nCAS2], uv, verb) + nCAS1
    return np.hstack((rid_ac1, rid_ac2))
#################################################


#################################################
def orbital_reorder_circ(mol, orbs, pts, method='angle', anchor=None, verb=4):
    '''
    pts:
      A 3x3 matrix where each row represents a Cartesian coordinate. The three
      points define the ordering plane.
    method:
      Ordering method, the options are "angle" (default) and "dipole". For "angle",
      the orbitals are ordered based on the cosine angle of their dipole vectors
      relative to the dipole vector of the anchor orbital. When "dipole" is chosen,
      the ordering is based on the x and y components of the dipole vectors. Here,
      the x and y directions are intrinsic coordinate defined for the ordering 
      plane and are based on the direction of the largest magnitude dipole vector.
      All dipole vectors are measured relative to the geometric center.
    anchor:
      A tuple of two elements, the first one being either "first" or "last", and
      the second one being a 1-base integer that specifies the orbital index to be
      used as the anchor. "first" ("last") means that the anchor orbitals is placed 
      at the left (right) most site. If not specified, the anchor is taken to be the 
      orbitals having the largest dipole vector magnitude.
    '''

    #assert drc == 'cw' or drc == 'ccw'
    if anchor is not None: assert len(anchor) == 2 and anchor[1] >= 1
    
    #==== Construct two vectors defining the orbitals ordering plane ====#
    a = pts[0,:] - pts[2,:]
    b = pts[1,:] - pts[2,:]

    #==== Define new z axis ====#
    uz_new = np.cross(a,b)
    uz_new = uz_new / np.linalg.norm(uz_new)
    
    #==== Dipole vectors ====#
    dip_ao = mol.intor('int1e_r').reshape(3,mol.nao,mol.nao)
    dip = np.einsum('ji, xjk, ki -> xi', orbs, dip_ao, orbs)   # 1)
    proj = np.eye(3) - np.outer(uz_new, uz_new)
    dip = proj @ dip                                           # 2)
    # 1) dip = actual dipole vectors w.r.t. the origin.
    # 2) dip = dipole vectors w.r.t the origin projected onto the ordering plane.

    #==== Geometric center of the dipole vectors system ====#
    gc = np.sum(dip, axis=1) / dip.shape[1]                  # 3)
    if verb >= 2:
        print('Geometric center = ' + '  '.join('%.5f ' % c for c in gc))
    for i in range(dip.shape[1]): dip[:,i] = dip[:,i] - gc   # 4)
    # 3) gc = position of the geometric center of the projected dipole vectors.
    # 4) dip = dipole vectors w.r.t the geom. center.

    #==== Ordering method ====#
    nn = np.linalg.norm(dip, axis=0)
    id_maxdip = np.argmax(nn)
    # By default, the anchor is the orbital having the largest dipole magnitude
    # relative to the geom. center.
    if anchor is None:
        anchor0 = id_maxdip
    else:
        anchor0 = anchor[1] - 1
    if verb >= 2:
        print('Largest dipole magnitude index (default anchor) = ', id_maxdip)
        print('Anchor index = ', anchor0)

    #==== New x and y axes ====#
    ux_new = dip[:,id_maxdip] / np.linalg.norm(dip[:,id_maxdip])
    uy_new = np.cross(uz_new, ux_new)
    if verb >= 2:
        print('Ordering plane x-axis = ' + ' '.join('%.4f'%c for c in ux_new))
        print('Ordering plane y-axis = ' + ' '.join('%.4f'%c for c in uy_new))
        print('Ordering plane z-axis = ' + ' '.join('%.4f'%c for c in uz_new))
    # The new x axis is defined to be the dipole vector of the orbital with the
    # largest dipole magnitude relative to geometric center. Then the new y axis
    # is equal to z_new x x_new.
    
    #==== Projection of dipole vectors onto the new x and y axes ====#
    x_new_proj = ux_new @ dip
    y_new_proj = uy_new @ dip
    
    if method == 'angle':
        agl = np.zeros(dip.shape[1])
        for j in range(len(agl)):
            t = np.dot(dip[:,id_maxdip],dip[:,j])/\
                np.linalg.norm(dip[:,id_maxdip])/\
                np.linalg.norm(dip[:,j])
            if abs(t) > 1.0:
                if abs(t)-1.0 <= 1E-13:
                    t = t/abs(t)
                else:
                    raise ValueError(f'The relative angle at j = {j} ' +
                                     'is larger than 1')
            agl[j] = np.arccos(t)

        #==== Sort the lower half of new y axis in decreasing x comp. ====#
        e1 = agl[y_new_proj <= 0.0]
        id1_ = np.where(y_new_proj <= 0.0)[0]
        id1 = id1_[np.argsort(e1)]
    
        #==== Sort the lower half of new y axis in increasing x comp. ====#
        e2 = agl[y_new_proj > 0.0]
        id2_ = np.where(y_new_proj > 0.0)[0]
        id2 = id2_[np.argsort(-e2)]

    elif method == 'dipole':
        #==== Sort the lower half of new y axis in decreasing x comp. ====#
        e1 = x_new_proj[y_new_proj <= 0.0]
        id1_ = np.where(y_new_proj <= 0.0)[0]
        id1 = id1_[np.argsort(-e1)]
    
        #==== Sort the lower half of new y axis in increasing x comp. ====#
        e2 = x_new_proj[y_new_proj > 0.0]
        id2_ = np.where(y_new_proj > 0.0)[0]
        id2 = id2_[np.argsort(e2)]

    #==== Construct the ordering vector ====#
    order_id = np.hstack((id1, id2))
    assert len(order_id) == orbs.shape[1]
    anchor_pos = order_id.tolist().index(anchor0)
    if anchor is not None:
        if anchor[0] == 'first':
            order_id = np.roll(order_id, -anchor_pos)
        elif anchor[0] == 'last':
            order_id = np.roll(order_id, len(order_id)-(anchor_pos+1))
        else:
            raise ValueError('Bad value of anchor[0]. It must be either ' + 
                             '"first" or "last".')

    #OLD#==== CW or CCW as seen from origin ====#
    #OLDflip = (np.dot(uz_new,gc) >= 0.0 and drc == 'cw' or
    #OLD        np.dot(uz_new,gc) < 0.0 and drc == 'ccw')
    #OLDif flip: order_id = order_id[::-1]
    if verb >= 2:
        print('Circular ordering indexes = ', order_id)

    #==== Print analysis ====#
    if verb >= 2:
        osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, orbs)
        print('%3s  %10s  %9s %9s %9s   %9s' %
                  ('No.', 'Rel. angle', 'Ord. x', 'Ord. y', 'Ord. z',
                   'Magn.'))
        for j in order_id:
            t = np.dot(dip[:,anchor0],dip[:,j])/\
                np.linalg.norm(dip[:,anchor0])/\
                np.linalg.norm(dip[:,j])
            if abs(t) > 1.0:
                if abs(t)-1.0 <= 1E-13:
                    t = t/abs(t)
                else:
                    raise ValueError(f'The relative angle at j = {j} ' +
                                     'is larger than 1')
            agl0 = np.arccos(t)
            print('%3d) %10.4f  %9.4f %9.4f %9.4f   %9.4f  %5s' %
                  (j+1, np.degrees(agl0), dip[0,j], dip[1,j], dip[2,j],
                   np.linalg.norm(dip[:,j]), osym[j]))
            
    return order_id
#################################################


#################################################
def orbital_reorder_mrci_circ():
    raise NotImplementedError('The function orbital_reorder_mrci_circ is' + 
                              'not implemented yet.')
#################################################


#################################################
def get_orb_sym(mol, orb):
    
    sym = []
    for i in range(0, orb.shape[1]):
        try:
            sym += [symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                        orb[:,i:i+1])[0]]
        except:
            sym += ['UNSYM']

    return sym
#################################################
