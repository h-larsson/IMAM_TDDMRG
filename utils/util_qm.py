import numpy as np
from block2 import SU2, SZ
from IMAM_TDDMRG.utils.util_print import _print

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
def get_symCASCI_ints(mol, nCore, nCAS, nelCAS, ocoeff, verbose):
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
        _print('WARNING: SZ Spin label is chosen! The get_symCASCI_ints function was ' +
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


    #==== Get the CAS orbitals and wavefunction symmetries ====#
    wSym = _mcCI.fcisolver.wfnsym
    wSym = wSym if wSym is not None else 0
    molpro_wSym = pyscf_tools.fcidump.ORBSYM_MAP[mol.groupname][wSym]
    oSym = np.array(_mcCI.mo_coeff.orbsym)[nCore:nCore+nCAS]
    #debug _print('here osym = ', oSym)
    molpro_oSym = [pyscf_tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in oSym]
    #debug _print('here molpro_osym = ', molpro_oSym)
    #forlater irname = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, _mcCI.mo_coeff)
    #forlater _print('here mo_coeff irname = ', irname)


    #==== Get the 1e and 2e integrals ====#
    h1e, eCore = _mcCI.get_h1cas()
    g2e = _mcCI.get_h2cas()
    g2e = np.require(g2e, dtype=np.float64)
    h1e = np.require(h1e, dtype=np.float64)


    #==== Some checks ====#
    assert oSym.size == nCAS, f'nCAS={nCAS} vs. oSym.size={oSym.size}'
    assert wSym == 0, "Want A1g state"      # IMAM: Why does it have to be A1g?
    
    del _mcCI, mf

    return h1e, g2e, eCore, molpro_oSym, molpro_wSym
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
