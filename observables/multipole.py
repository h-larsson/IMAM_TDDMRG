import numpy as np


def calc(mol, dpole_ao, qpole_ao, pdm, mo):
    '''
    Calculates the expectation value of a spin-independent 1-electron operator 
    (whose first quantization form is O = o(1) + o(2) + ... + o(N) for an N-electron 
    system) using the 1pdm.
    '''

    #==== Complex or real PDM? ====#
    assert len(pdm.shape) == 3, 'multipole.calc: pdm is not a 3D array.'
    assert pdm.shape[1] == pdm.shape[2], 'multipole.calc: pdm is not square.'
    
    #==== Dipole ====#
    n_dpole = np.zeros((3))
    for i in range(0,mol.natm):
        n_dpole += mol.atom_charge(i) * mol.atom_coord(i)
    dpole_mo = np.einsum('sji,xjk,skl -> xsil', mo, dpole_ao, mo)
    e_dpole = -np.einsum('xskj,sjk -> x', dpole_mo, pdm)

    #==== Quadrupole ====#
    n_qpole = np.zeros((3,3))
    for i in range(0,mol.natm):
        n_qpole += mol.atom_charge(i) * np.outer(mol.atom_coord(i), mol.atom_coord(i))
    qpole_mo = np.einsum('sji,xyjk,skl -> xysil', mo, qpole_ao, mo)
    e_qpole = -np.einsum('xyskj,sjk -> xy', qpole_mo, pdm)

    return e_dpole, n_dpole, e_qpole, n_qpole
    #################################################
