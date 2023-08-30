#OLD_CPX import numpy as np
#OLD_CPX from block2 import VectorUBond, VectorDouble, OpNamesSet, NoiseTypes, DecompositionTypes
#OLD_CPX from block2 import SU2, SZ, EquationTypes
#OLD_CPX 
#OLD_CPX # Set spin-adapted or non-spin-adapted here
#OLD_CPX SpinLabel = SU2
#OLD_CPX #SpinLabel = SZ
#OLD_CPX 
#OLD_CPX if SpinLabel == SU2:
#OLD_CPX     from block2.su2 import MovingEnvironment, Linear
#OLD_CPX     from block2.su2 import Expect
#OLD_CPX     try:
#OLD_CPX         from block2.su2 import MPICommunicator
#OLD_CPX         hasMPI = True
#OLD_CPX     except ImportError:
#OLD_CPX         hasMPI = False
#OLD_CPX else:
#OLD_CPX     from block2.sz import MovingEnvironment, Linear
#OLD_CPX     from block2.sz import Expect
#OLD_CPX     try:
#OLD_CPX         from block2.sz import MPICommunicator
#OLD_CPX         hasMPI = True
#OLD_CPX     except ImportError:
#OLD_CPX         hasMPI = False


import numpy as np
import subprocess, shutil, os
from IMAM_TDDMRG.utils.util_complex_type import get_complex_type


spin_symmetry = 'su2'
#spin_symmetry = 'sz'

comp = get_complex_type()
import block2 as b2
if comp == 'full':
    bx = b2.cpx
    bc = bx
elif comp == 'hybrid':
    bx = b2
    bc = None    #OLD block2.cpx if has_cpx else None

if spin_symmetry == 'su2':
    bs = bx.su2
    brs = b2.su2
    SX = b2.SU2
elif spin_symmetry == 'sz':
    bs = bx.sz
    brs = b2.sz
    SX = b2.SZ

try:
    if spin_symmetry == 'su2':
        from block2.su2 import MPICommunicator
    elif spin_symmetry == 'sz':
        from block2.sz import MPICommunicator
    hasMPI = True
except ImportError:
    MPICommunicator = ParallelRuleIdentity = None
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
        lme = bs.MovingEnvironment(lmpo, fitket, fitket, "PERT")
        lme.init_environments(False)
        if delay_contract:
            lme.delayed_contraction = b2.OpNamesSet.normal_ops()
    #fordebug rme = MovingEnvironment(lmpo, mps, mps, "RHS")
    rme = bs.MovingEnvironment(rmpo, fitket, mps, "RHS")
    rme.init_environments(False)
    if delay_contract:
        rme.delayed_contraction = b2.OpNamesSet.normal_ops()
        
    #==== Begin MPS fitting ====#
    if fit_margin == None:
        fit_margin = max(int(mps.info.bond_dim / 10.0), 100)
    fit = bs.Linear(lme, rme, b2.VectorUBond(fit_bond_dims),
                    b2.VectorUBond([mps.info.bond_dim + fit_margin]),
                    b2.VectorDouble(fit_noises))
    
    if noise_type == 'reduced_perturb':
        fit.noise_type = b2.NoiseTypes.ReducedPerturbative
    elif noise_type == 'reduced_perturb_lowmem':
        fit.noise_type = b2.NoiseTypes.ReducedPerturbativeCollectedLowMem
    elif noise_type == 'density_mat':
        fit.noise_type = b2.NoiseTypes.DensityMatrix
    else:
        raise ValueError("The 'noise_type' parameter of 'MPS_fitting' does not" +
                         "correspond to any available options, which are 'reduced_perturb', " +
                         "'reduced_perturb_lowmem', or 'density_mat'.")
    
    if decomp_type == 'svd':
        fit.decomp_type = b2.DecompositionTypes.SVD
    elif decomp_type == 'density_mat':
        fit.decomp_type = b2.DecompositionTypes.DensityMatrix
    else:
        raise ValueError("The 'decomp_type' parameter of 'MPS_fitting' does not" +
                         "correspond to any available options, which are 'svd' or 'density_mat'.")

    if lme is not None:
        fit.eq_type = b2.EquationTypes.PerturbativeCompression
    fit.iprint = max(verbose_lvl, 0)
    fit.cutoff = cutoff
    fit.solve(fit_nsteps, mps.center == 0, fit_conv_tol)
#################################################


#################################################
def calc_energy_MPS(hmpo, mps, bond_dim_margin=0):

    me = bs.MovingEnvironment(hmpo, mps, mps, "me_erg")
    me.init_environments(False)
    D = mps.info.bond_dim + bond_dim_margin
    expect = bs.Expect(me, D, D)
    erg = expect.solve(False, mps.center == 0)

    return erg
#################################################


#################################################
def loadMPSfromDir(mps_info: brs.MPSInfo,  mpsSaveDir:str, MPI:MPICommunicator=None) -> bs.MPS:
    """  Load MPS from directory
    :param mps_info: If None, MPSInfo will be read from mpsSaveDir/mps_info.bin
    :param mpsSaveDir: Directory where the MPS has been saved
    :param MPI: MPI class (or None)
    :return: MPS state
    """
    if mps_info is None: # use mps_info.bin
        mps_info = brs.MPSInfo(0)
        mps_info.load_data(f"{mpsSaveDir}/mps_info.bin")
    else:
        # TODO: It would be good to check if mps_info.bin is available
        #       and then compare it again mps_info input to see any mismatch
        pass 
    def copyItRev(fnam: str):
        if MPI is not None and MPI.rank != 0:
            # ATTENTION: For multi node  calcs, I assume that all nodes have one global scratch dir
            return
        lastName = os.path.split(fnam)[-1]
        fst = f"cp -p {mpsSaveDir}/{lastName} {fnam}"
        try:
            subprocess.call(fst.split())
        except: # May problem due to allocate memory, but why??? 
            print(f"# ATTENTION: loadMPSfromDir with command'{fst}' failed!")
            print(f"# Error message: {sys.exc_info()[0]}")
            print(f"# Error message: {sys.exc_info()[1]}")
            print(f"# Try again with shutil")
            try:
                # vv does not copy metadata -.-
                shutil.copyfile(mpsSaveDir+"/"+lastName, fnam)
            except:
                print(f"\t# ATTENTION: loadMPSfromDir with shutil also failed")
                print(f"\t# Error message: {sys.exc_info()[0]}")
                print(f"\t# Error message: {sys.exc_info()[1]}")
                print(f"\t# Try again with syscal")
                os.system(fst)
    for iSite in range(mps_info.n_sites + 1):
        fnam = mps_info.get_filename(False, iSite)
        copyItRev(fnam)
        if MPI is not None:
            MPI.barrier()
        fnam = mps_info.get_filename(True, iSite)
        copyItRev(fnam)
        if MPI is not None:
            MPI.barrier()
    mps_info.load_mutable()
    mps = bs.MPS(mps_info)
    for iSite in range(-1, mps_info.n_sites):  # -1 is data
        fnam = mps.get_filename(iSite)
        copyItRev(fnam)
        if MPI is not None:
            MPI.barrier()
    mps.load_data()
    mps.load_mutable()
    if MPI is not None:
        MPI.barrier()
    mps_info.bond_dim = mps.info.get_max_bond_dimension() # is not initalized
    return mps
#################################################
