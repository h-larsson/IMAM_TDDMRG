import numpy as np
from block2 import VectorUBond, VectorDouble, OpNamesSet, NoiseTypes, DecompositionTypes
from block2 import SU2, SZ, EquationTypes

# Set spin-adapted or non-spin-adapted here
SpinLabel = SU2
#SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import MovingEnvironment, Linear
    from block2.su2 import Expect
    try:
        from block2.su2 import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False
else:
    from block2.sz import MovingEnvironment, Linear
    from block2.sz import Expect
    try:
        from block2.sz import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False

from IMAM_TDDMRG.utils.util_print import getVerbosePrinter    
if hasMPI:
    MPI = MPICommunicator()
    r0 = (MPI.rank == 0)
else:
    class _MPI:
        rank = 0
    MPI = _MPI()
    r0 = True
    
_print = getVerbosePrinter(r0, flush=True)
print_i2 = getVerbosePrinter(r0, indent=2*' ', flush=True)
print_i4 = getVerbosePrinter(r0, indent=4*' ', flush=True)    





#################################################
def print_MPO_bond_dims(mpo, name=''):
    mpo_bdims = [None] * len(mpo.left_operator_names)
    for ix in range(len(mpo.left_operator_names)):
        mpo.load_left_operators(ix)
        x = mpo.left_operator_names[ix]
        mpo_bdims[ix] = x.m * x.n
        mpo.unload_left_operators(ix)
    _print(name + ' MPO BOND DIMS = ', ''.join(["%6d" % x for x in mpo_bdims]))
#################################################


#################################################
def MPS_fitting(fitket, mps, rmpo, fit_bond_dims, fit_nsteps, fit_noises, 
                fit_conv_tol, decomp_type, cutoff, lmpo=None, fit_margin=None, 
                noise_type='reduced_perturb', delay_contract=True, verbose_lvl=1):

    #==== Construct the LHS and RHS Moving Environment objects ====#
    if lmpo is None:
        lme = None
    else:
        lme = MovingEnvironment(lmpo, fitket, fitket, "PERT")
        lme.init_environments(False)
        if delay_contract:
            lme.delayed_contraction = OpNamesSet.normal_ops()
    #fordebug rme = MovingEnvironment(lmpo, mps, mps, "RHS")
    rme = MovingEnvironment(rmpo, fitket, mps, "RHS")
    rme.init_environments(False)
    if delay_contract:
        rme.delayed_contraction = OpNamesSet.normal_ops()

        
    #==== Begin MPS fitting ====#
    if fit_margin == None:
        fit_margin = max(int(mps.info.bond_dim / 10.0), 100)
    fit = Linear(lme, rme, VectorUBond(fit_bond_dims),
                 VectorUBond([mps.info.bond_dim + fit_margin]), VectorDouble(fit_noises))
    
    if noise_type == 'reduced_perturb':
        fit.noise_type = NoiseTypes.ReducedPerturbative
    elif noise_type == 'reduced_perturb_lowmem':
        fit.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
    elif noise_type == 'density_mat':
        fit.noise_type = NoiseTypes.DensityMatrix
    else:
        raise ValueError("The 'noise_type' parameter of 'MPS_fitting' does not" +
                         "correspond to any available options, which are 'reduced_perturb', " +
                         "'reduced_perturb_lowmem', or 'density_mat'.")
    
    if decomp_type == 'svd':
        fit.decomp_type = DecompositionTypes.SVD
    elif decomp_type == 'density_mat':
        fit.decomp_type = DecompositionTypes.DensityMatrix
    else:
        raise ValueError("The 'decomp_type' parameter of 'MPS_fitting' does not" +
                         "correspond to any available options, which are 'svd' or 'density_mat'.")

    if lme is not None:
        fit.eq_type = EquationTypes.PerturbativeCompression
    fit.iprint = max(verbose_lvl, 0)
    fit.cutoff = cutoff
    fit.solve(fit_nsteps, mps.center == 0, fit_conv_tol)
#################################################


#################################################
def calc_energy_MPS(hmpo, mps, bond_dim_margin=0):

    me = MovingEnvironment(hmpo, mps, mps, "me_erg")
    me.init_environments(False)
    D = mps.info.bond_dim + bond_dim_margin
    expect = Expect(me, D, D)
    erg = expect.solve(False, mps.center == 0)

    return erg
#################################################
