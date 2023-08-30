
import numpy as np
from scipy.linalg import eigh
from IMAM_TDDMRG.utils.util_print import _print

##########################################################
def get_atom_range(mol):
    nao = mol.nao
    ao_range = [None] * mol.natm
    start_found = False
    ia = 0
    for ia in range(0, mol.natm):
        for ib in range(0, nao):
            ibm = min(ib+1, nao-1)
            if mol.ao_labels(fmt=False)[ib][0] == ia:
                if not start_found:
                    ia_start = ib
                    start_found = True
                if mol.ao_labels(fmt=False)[ibm][0] == ia+1 or ib == nao-1:
                    ia_last = ib
                    start_found = False
                    break
        ao_range[ia] = (ia_start, ia_last)

    return ao_range
##########################################################


##########################################################
def calc(mol, pdm, mo, ovl=None):
    '''
    mol = Mole object.
    pdm = The complete (core+active) one-particle-reduced density matrix in MO rep.
    mo = The MOs in AO rep.
    ovl = The overlap matrix associated with the AO basis defined in mol.
    '''    


    #==== Complex or real PDM? ====#
    assert len(pdm.shape) == 3, 'partial_charge.calc: pdm is not a 3D array.'
    assert len(mo.shape) == 3, 'partial_charge.calc: mo is not a 3D array.'
    assert pdm.shape[1] == pdm.shape[2], 'partial_charge.calc: pdm is not square.'
    complex_pdm = (type(pdm[0,0,0]) == np.complex128)
    dtype = np.complex128 if complex_pdm else np.float64

    
    #==== Setting up the system ====#
    nao = mol.nao
    atom_ao_range = get_atom_range(mol)

    
    #==== AO overlap matrix ====#
    if ovl is None:
        ovl = mol.intor('int1e_ovlp')
    es, U = eigh(ovl)
    ovl_half = U @ (np.diag( np.sqrt(es) ) @ U.conj().T)
    
    
    #==== Calculate the partial charge ====#
    P = np.zeros((nao, nao), dtype=dtype)
    for i in range(0,2):
        P = P + mo[i,:,:] @ (pdm[i,:,:] @ mo[i,:,:].T)
    
    Tmul = np.einsum('ij, ji -> i', P, ovl)
    qmul = [None] * mol.natm
    Tlow = np.einsum('ij, jk, ki -> i', ovl_half, P, ovl_half)
    qlow = [None] * mol.natm
    for ia in range(0, mol.natm):
        ib_1 = atom_ao_range[ia][0]
        ib_2 = atom_ao_range[ia][1]
    
        #==== Mulliken population ====#
        qmul[ia] = mol.atom_charge(ia) - np.sum( Tmul[ib_1:ib_2+1] )
    
        #==== Lowdin population ====#
        qlow[ia] = mol.atom_charge(ia) - np.sum( Tlow[ib_1:ib_2+1] )

    return qmul, qlow
##########################################################
