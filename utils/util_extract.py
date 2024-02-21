from pyscf import gto
from IMAM_TDDMRG.utils.util_atoms import get_tot_nuc_charge, extract_atoms



def mole(inputs):
    nelCore = 2 * inputs['nCore']
    tot_nq = get_tot_nuc_charge(extract_atoms(inputs['inp_coordinates']))
    charge = tot_nq - nelCore - inputs['nelCAS']
    mol = gto.M(atom=inputs['inp_coordinates'], basis=inputs['inp_basis'],
                ecp=inputs['inp_ecp'], symmetry=inputs['inp_symmetry'],
                charge=charge, spin=inputs['twos'])
    return mol
