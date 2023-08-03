import numpy as np
from block2 import SU2, SZ

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
