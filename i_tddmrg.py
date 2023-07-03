
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#

"""
A class for the analysis of charge migration starting from
an initial state produced by the application of the annihilation 
operator on a given site/orbital.

Original version:
     Imam Wahyutama, May 2023
Derived from:
     gfdmrg.py
     ft_tddmrg.py
"""

#from ipsh import *
from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16, TETypes
import time
import numpy as np
import scipy.linalg
from scipy.linalg import eigvalsh

# Set spin-adapted or non-spin-adapted here
SpinLabel = SU2
#SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
    from block2.su2 import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
    try:
        from block2.su2 import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False
else:
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, Linear, IdentityMPO
    from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
    from block2.sz import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
    try:
        from block2.sz import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False


import tools; tools.init(SpinLabel)
from tools import saveMPStoDir, loadMPSfromDir, mkDir
from gfdmrg import orbital_reorder

        
if hasMPI:
    MPI = MPICommunicator()
else:
    class _MPI:
        rank = 0
    MPI = _MPI()
    
    

#################################################
def _print(*args, **kwargs):
    if MPI.rank == 0:
        print(*args, **kwargs)
#################################################


#################################################
def printDummyFunction(*args, **kwargs):
    """ Does nothing"""
    pass
#################################################


#################################################
def getVerbosePrinter(verbose,indent="",flush=False):
    if verbose:
        if flush:
            def _print(*args, **kwargs):
                kwargs["flush"] = True
                print(indent,*args,**kwargs)
        else:
            def _print(*args, **kwargs):
                print(indent, *args, **kwargs)
    else:
        _print = printDummyFunction
    return _print        
#################################################


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
    molpro_oSym = [pyscf_tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in oSym]   


    #==== Get the 1e and 2e integrals ====#
    h1e, eCore = _mcCI.get_h1cas()
    g2e = _mcCI.get_h2cas()
    g2e = np.require(g2e, dtype=np.float64)
    h1e = np.require(h1e, dtype=np.float64)
    #move_out    g2eShape = g2e.shape
    #move_out    h1eShape = h1e.shape


    #==== Some checks ====#
    assert oSym.size == nCAS, f'nCAS={nCAS} vs. oSym.size={oSym.size}'
    assert wSym == 0, "Want A1g state"      # IMAM: Why does it have to be A1g?
    #OLD if verbose:
    #OLD     _print("# oSym = ", oSym)
    #OLD     _print("# oSym = ", [symm.irrep_id2name(mol.groupname,s) for s in oSym])
    #OLD     _print("# wSym = ", wSym, symm.irrep_id2name(mol.groupname,wSym), flush=True)
    
    
    del _mcCI, mf

    return h1e, g2e, eCore, molpro_oSym, molpro_wSym
#################################################



#################################################
class MYTDDMRG:
    """
    DDMRG++ for Green's Function for molecules.
    """


    #################################################
    def __init__(self, scratch='./nodex', memory=1*1E9, isize=2E8, omp_threads=8, verbose=2,
                 print_statistics=True, mpi=None, delayed_contraction=True):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """


        if SpinLabel == SZ:
            _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            _print('WARNING: SZ Spin label is chosen! The MYTDDMRG class was designed ' +
                   'with the SU2 spin label in mind. The use of SZ spin label in this ' +
                   'class has not been checked.')
            _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        _print('Memory = %10.2f Megabytes' % (memory/1.0e6))
        _print('Integer size = %10.2f Megabytes' % (isize/1.0e6))

        
        Random.rand_seed(0)
        isize = int(isize)
        assert isize < memory
        #OLD isize = min(int(memory * 0.1), 200000000)
        init_memory(isize=isize, dsize=int(memory - isize), save_dir=scratch)
        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
        Global.threading.seq_type = SeqTypes.Tasked
        Global.frame.load_buffering = False
        Global.frame.save_buffering = False
        Global.frame.use_main_stack = False
        Global.frame.minimal_disk_usage = True

        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.print_statistics = print_statistics
        self.mpi = mpi
        ## self.mpi = MPI
        
        self.delayed_contraction = delayed_contraction
        self.idx = None # reorder
        self.ridx = None # inv reorder
        if self.mpi is not None:
            print('herey2 I am MPI', self.mpi.rank)
            self.mpi.barrier()

        
        #==== Create scratch directory ====#
        if self.mpi is not None:
            if self.mpi.rank == 0:
                mkDir(scratch)
            self.mpi.barrier()
        else:
            mkDir(scratch)

        if self.verbose >= 2:
            _print(Global.frame)
            _print(Global.threading)

        if mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelRuleSiteQC
                from block2.su2 import ParallelRuleSiteQC, ParallelRuleIdentity
            else:
                from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
                from block2.sz import ParallelRuleSiteQC, ParallelRuleIdentity
            self.prule = ParallelRuleQC(mpi)
            self.pdmrule = ParallelRuleNPDMQC(mpi)
            self.siterule = ParallelRuleSiteQC(mpi)
            self.identrule = ParallelRuleIdentity(mpi)
        else:
            self.prule = None
            self.pdmrule = None
            self.siterule = None
            self.identrule = None
    #################################################

            
    #################################################
    def init_hamiltonian_fcidump(self, pg, filename, idx=None):
        """Read integrals from FCIDUMP file."""
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)
        #OLD self.orb_sym = VectorUInt8(
        #OLD     map(PointGroup.swap_d2h, self.fcidump.orb_sym))
        swap_pg = getattr(PointGroup, "swap_" + pg)
        self.orb_sym = VectorUInt8(map(swap_pg, self.fcidump.orb_sym))
        _print("# fcidump symmetrize error:", self.fcidump.symmetrize(orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                swap_pg(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
    #################################################


    #################################################
    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym,
                         e_core, h1e, g2e, tol=1E-13, idx=None,
                         save_fcidump=None):
        """
        Initialize integrals using h1e, g2e, etc.
        isym: wfn symmetry in molpro convention? See the getFCIDUMP function in CAS_example.py.
        g2e: Does it need to be in 8-fold symmetry? See the getFCIDUMP function in CAS_example.py.
        orb_sym: orbitals symmetry in molpro convention.
        """

        #==== Initialize self.fcidump ====#
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        if not isinstance(h1e, tuple):
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol, '\n' + \
                        f'   h1e[i,j] = {h1e[i,j]:17.13f} \n' + \
                        f'   h1e[j,i] = {h1e[j,i]:17.13f} \n' + \
                        f'   Delta = {h1e[i, j] - h1e[j, i]:17.13f} \n' + \
                        f'   tol. = {tol:17.13f}'
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.ravel()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
            if self.verbose >= 2:
                _print('Number of 1e integrals (incl. hermiticity) = ', mh1e.size)
                _print('Number of 2e integrals (incl. hermiticity) = ', mg2e.size)
            
            self.fcidump.initialize_su2(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        else:
            assert SpinLabel == SZ
            assert isinstance(h1e, tuple) and len(h1e) == 2
            assert isinstance(g2e, tuple) and len(g2e) == 3
            mh1e_a = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e_b = np.zeros((n_sites * (n_sites + 1) // 2))
            mh1e = (mh1e_a, mh1e_b)
            for xmh1e, xh1e in zip(mh1e, h1e):
                k = 0
                for i in range(0, n_sites):
                    for j in range(0, i + 1):
                        assert abs(xh1e[i, j] - xh1e[j, i]) < tol
                        xmh1e[k] = xh1e[i, j]
                        k += 1
                xmh1e[np.abs(xmh1e) < tol] = 0.0
#OLD            mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            mg2e = tuple(xg2e.ravel() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0      # xmg2e works like a pointer to the elements of mg2e tuple.
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)


        #==== Take care of the symmetry conventions. Note that ====#
        #====   self.fcidump.orb_sym is in Molpro convention,  ====#
        #====     while self.orb_sym is in block2 convetion    ====#
        self.fcidump.orb_sym = VectorUInt8(orb_sym)       # Hence, self.fcidump.orb_sym is in Molpro convention.

        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)
#OLD        self.orb_sym = VectorUInt8(
#OLD            map(PointGroup.swap_d2h, self.fcidump.orb_sym))
        swap_pg = getattr(PointGroup, "swap_" + pg)
        self.orb_sym = VectorUInt8(map(swap_pg, self.fcidump.orb_sym))


        #==== Construct the Hamiltonian MPO ====#
        vacuum = SpinLabel(0)
        self.target = SpinLabel(n_elec, twos, swap_pg(isym))
        self.n_sites = n_sites
        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        
        #==== Save self.fcidump ====#
        if save_fcidump is not None:
            if self.mpi is None or self.mpi.rank == 0:
                self.fcidump.orb_sym = VectorUInt8(orb_sym)
                self.fcidump.write(save_fcidump)
            if self.mpi is not None:
                self.mpi.barrier()

        if self.mpi is not None:
            self.mpi.barrier()
    #################################################


    #################################################
    @staticmethod
    def fmt_size(i, suffix='B'):
        if i < 1000:
            return "%d %s" % (i, suffix)
        else:
            a = 1024
            for pf in "KMGTPEZY":
                p = 2
                for k in [10, 100, 1000]:
                    if i < k * a:
                        return "%%.%df %%s%%s" % p % (i / a, pf, suffix)
                    p -= 1
                a *= 1024
        return "??? " + suffix
    #################################################


    #################################################
    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, iscomp, mps=None, inmps_name=None, dmargin=0):
        if mps is None and inmps_name is None:
            raise ValueError("The 'mps' and 'inmps_name' parameters of "
                             + "get_one_pdm cannot be both None.")
        
        if self.verbose >= 2:
            _print('>>> START one-pdm <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        if mps is None:   # mps takes priority over inmps_name, the latter will only be used if the former is None.
            mps_info = MPSInfo(0)
            mps_info.load_data(self.scratch + "/" + inmps_name)
            mps = MPS(mps_info)
            mps.load_data()
            mps.info.load_mutable()
            
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        # 1PDM MPO
        pmpo = PDM1MPOQC(self.hamil)
        pmpo = SimplifiedMPO(pmpo, RuleQC())
        if self.mpi is not None:
            pmpo = ParallelMPO(pmpo, self.pdmrule)

        # 1PDM
        pme = MovingEnvironment(pmpo, mps, mps, "1PDM")
        pme.init_environments(False)
        if iscomp:
            expect = ComplexExpect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
        else:
            expect = Expect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, mps.center == 0)
        if SpinLabel == SU2:
            dmr = expect.get_1pdm_spatial(self.n_sites)
            dm = np.array(dmr).copy()
        else:
            dmr = expect.get_1pdm(self.n_sites)
            dm = np.array(dmr).copy()
            dm = dm.reshape((self.n_sites, 2,
                             self.n_sites, 2))
            dm = np.transpose(dm, (0, 2, 1, 3))

        if self.ridx is not None:
            dm[:, :] = dm[self.ridx, :][:, self.ridx]

        mps.save_data()
        if mps is None:
            mps_info.deallocate()
        dmr.deallocate()
        pmpo.deallocate()

        if self.verbose >= 2:
            _print('>>> COMPLETE one-pdm | Time = %.2f <<<' %
                   (time.perf_counter() - t))

        if SpinLabel == SU2:
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        else:
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
    #################################################

    
    #################################################
    def dmrg(self, bond_dims, noises, n_steps=30, dav_tols=1E-5, conv_tol=1E-7, cutoff=1E-14,
             occs=None, bias=1.0, outmps_dir0=None, outmps_name='GS_MPS_INFO',
             save_1pdm=False):
        """Ground-State DMRG."""

        if self.verbose >= 2:
            _print('>>> START GS-DMRG <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()
        if outmps_dir0 is None:
            outmps_dir = self.scratch
        else:
            outmps_dir = outmps_dir0

        # MultiMPSInfo
        mps_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                           self.target, self.hamil.basis)
        mps_info.tag = 'KET'
        if occs is None:
            if self.verbose >= 2:
                _print("Using FCI INIT MPS")
            mps_info.set_bond_dimension(bond_dims[0])
        else:
            if self.verbose >= 2:
                _print("Using occupation number INIT MPS")
            if self.idx is not None:
                occs = self.fcidump.reorder(VectorDouble(occs), VectorUInt16(self.idx))
            mps_info.set_bond_dimension_using_occ(
                bond_dims[0], VectorDouble(occs), bias=bias)
        _print('herep1')
        mps = MPS(self.n_sites, 0, 2)   # The 3rd argument controls the use of one/two-site algorithm.
        mps.initialize(mps_info)
        mps.random_canonicalize()

        _print('herep2')
        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()
        

        _print('herep3')
        # MPO
        tx = time.perf_counter()
        _print('herep4')
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        _print('herep5')
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
        _print('herep6')
        self.mpo_orig = mpo

        _print('herep4')
        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
            mpo = ParallelMPO(mpo, self.prule)

        if self.verbose >= 3:
            _print('MPO time = ', time.perf_counter() - tx)

        if self.print_statistics:
            _print('GS MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("GS EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("GS EST PEAK MEM = ", MYTDDMRG.fmt_size(
                mem2), " SCRATCH = ", MYTDDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        _print('herep5')
        # DMRG
        me = MovingEnvironment(mpo, mps, mps, "DMRG")
        if self.delayed_contraction:
            me.delayed_contraction = OpNamesSet.normal_ops()
            me.cached_contraction = True
        tx = time.perf_counter()
        me.init_environments(self.verbose >= 4)
        if self.verbose >= 3:
            _print('DMRG INIT time = ', time.perf_counter() - tx)
        dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
        dmrg.davidson_conv_thrds = VectorDouble(dav_tols)
        dmrg.davidson_soft_max_iter = 4000
        dmrg.noise_type = NoiseTypes.ReducedPerturbative
        dmrg.decomp_type = DecompositionTypes.SVD
        dmrg.iprint = max(self.verbose - 1, 0)
        dmrg.cutoff = cutoff
        dmrg.solve(n_steps, mps.center == 0, conv_tol)

        self.gs_energy = dmrg.energies[-1][0]
        self.bond_dim = bond_dims[-1]
        dm0 = self.get_one_pdm(False, mps)
        _print('Molecular orbitals occupation: ')
        _print('   ')
        for i in range(0, self.n_sites):
            _print('%13.8f' % dm0[0, i, i], end=('\n' if i==self.n_sites-1 else ''))
        
        
        #==== Save the output MPS ====#
        #OLD mps.save_data()
        #OLD mps_info.save_data(self.scratch + "/GS_MPS_INFO")
        #OLD mps_info.deallocate()
        _print('Saving the ground state MPS files under ' + outmps_dir)
        if outmps_dir != self.scratch:
            mkDir(outmps_dir)
        mps_info.save_data(outmps_dir + "/" + outmps_name)
        saveMPStoDir(mps, outmps_dir, self.mpi)
        _print('Output ground state max. bond dimension = ', mps.info.bond_dim)
        if save_1pdm:
            np.save(outmps_dir + '/GS_1pdm', dm0)


        if self.print_statistics:
            dmain, dseco, imain, iseco = Global.frame.peak_used_memory
            _print("GS PEAK MEM USAGE:",
                   "DMEM = ", MYTDDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", MYTDDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        if self.verbose >= 1:
            _print("Ground state energy = %20.15f" % self.gs_energy)

        if self.verbose >= 2:
            _print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                   (time.perf_counter() - t))
    #################################################

    
    #################################################
    def save_gs_mps(self, save_dir='./gs_mps'):
        import shutil
        import pickle
        import os
        if self.mpi is None or self.mpi.rank == 0:
            pickle.dump(self.gs_energy, open(
                self.scratch + '/GS_ENERGY', 'wb'))
            for k in os.listdir(self.scratch):
                if '.KET.' in k or k == 'GS_MPS_INFO' or k == 'GS_ENERGY':
                    shutil.copy(self.scratch + "/" + k, save_dir + "/" + k)
        if self.mpi is not None:
            self.mpi.barrier()
    #################################################

    
    #################################################
    def load_gs_mps(self, load_dir='./gs_mps'):
        import shutil
        import pickle
        import os
        if self.mpi is None or self.mpi.rank == 0:
            for k in os.listdir(load_dir):
                shutil.copy(load_dir + "/" + k, self.scratch + "/" + k)
        if self.mpi is not None:
            self.mpi.barrier()
        self.gs_energy = pickle.load(open(self.scratch + '/GS_ENERGY', 'rb'))
    #################################################


    #################################################
    def print_occupation_table(self, dm, aid=None, mo_coeff=None):

        assert ((aid is None) ^ (mo_coeff is None)), 'Either aid or mo_coeff ' + \
               'should be None, but not both. '
        
        natocc_a = eigvalsh(dm[0,:,:])
        natocc_b = eigvalsh(dm[1,:,:])

        mk = ''
        if mo_coeff is None:
            mk = ' (ann.)'
        ll = 4 + (4 if mo_coeff is None else 5)*18 + 2*len(mk)
        hline = ''.join(['-' for i in range(0, ll)])
        aspace = ''.join([' ' for i in range(0,len(mk))])

        _print(hline)
        _print('%4s'  % 'No.', end='') 
        _print('%18s' % 'Alpha MO occ.' + aspace, end='')
        _print('%18s' % 'Beta MO occ.' + aspace,  end='')
        if mo_coeff is not None:
            _print('%18s' % 'ann. coeff', end='')
        _print('%18s' % 'Alpha natorb occ.', end='')
        _print('%18s' % 'Beta natorb occ.',  end='\n')
        _print(hline)

        for i in range(0, dm.shape[1]):
            if i == aid and mo_coeff is None:
                mk0 = mk
            else:
                mk0 = aspace

            _print('%4d'     % i, end='')
            _print('%18.8f' % np.diag(dm[0,:,:])[i] + mk0, end='')
            _print('%18.8f' % np.diag(dm[1,:,:])[i] + mk0, end='')
            if mo_coeff is not None:
                _print('%18.8f' % mo_coeff[i], end='')
            _print('%18.8f' % natocc_a[i], end='')
            _print('%18.8f' % natocc_b[i], end='\n')
    #################################################

    
    #################################################
    def annihilate(self, fit_bond_dims, fit_noises, fit_conv_tol, fit_n_steps, 
                   inmps_dir0=None, inmps_name='GS_MPS_INFO', outmps_dir0=None,
                   outmps_name='ANN_KET', aid=None, mo_coeff=None, cutoff=1E-14, alpha=True, 
                   occs=None, bias=1.0, outmps_normal=True, save_1pdm=False):
        """Green's function."""
        ##OLD ops = [None] * len(aid)
        ##OLD rkets = [None] * len(aid)
        ##OLD rmpos = [None] * len(aid)

        if self.mpi is not None:
            self.mpi.barrier()

        if inmps_dir0 is None:
            inmps_dir = self.scratch
        else:
            inmps_dir = inmps_dir0
        if outmps_dir0 is None:
            outmps_dir = self.scratch
        else:
            outmps_dir = outmps_dir0

        assert ((aid is None) ^ (mo_coeff is None)), 'Either aid or mo_coeff ' + \
            'should be None, but not both. '


        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

                
        #==== Build Hamiltonian MPO (to provide noise ====#
        #====      for Linear.solve down below)       ====#
        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo

        mpo = 1.0 * self.mpo_orig
        print_MPO_bond_dims(mpo, 'Hamiltonian')
        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)


        #==== Load the input MPS ====#
        inmps_path = inmps_dir + "/" + inmps_name
        _print('Loading input MPS info from ' + inmps_path)
        mps_info = MPSInfo(0)
        mps_info.load_data(inmps_path)
        #OLD mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        #OLD mps = MPS(mps_info)
        #OLD mps.load_data()
        mps = loadMPSfromDir(mps_info, inmps_dir, self.mpi)
        _print('Input MPS max. bond dimension = ', mps.info.bond_dim)
        dm0 = self.get_one_pdm(False, mps)
        _print('Occupations before annihilation:')
        self.print_occupation_table(dm0, aid, mo_coeff)

        
        #==== Some statistics ====#
        if self.print_statistics:
            max_d = max(fit_bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("EST MAX OUTPUT MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("EST PEAK MEM = ", MYTDDMRG.fmt_size(mem2),
                   " SCRATCH = ", MYTDDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

        #NOTE: check if ridx is not none

        
        #==== Begin constructing the annihilation MPO ====#
        if mo_coeff is None:
            idx = self.ridx[aid]
        else:
            if self.idx is not None:
                mo_coeff = mo_coeff[self.idx]            
            ops = [None] * self.n_sites

        gidxs = list(range(self.n_sites))
        for ii, ix in enumerate(gidxs):
            if mo_coeff is not None or (mo_coeff is None and ix==idx):
                if SpinLabel == SZ:
                    opsx = OpElement(OpNames.D, SiteIndex(
                        (ix, ), (0 if alpha else 1, )), SZ(-1, -1 if alpha else 1, self.orb_sym[ix]))
                else:
                    opsx = OpElement(OpNames.D, SiteIndex(
                        (ix, ), ()), SU2(-1, 1, self.orb_sym[ix]))
                if mo_coeff is not None:
                    ops[ii] = opsx
                else:
                    ops = opsx
                    

        #==== Determine if the annihilated orbital is a site orbital ====#
        #====  or a linear combination of them (orbital transform)   ====#
        if mo_coeff is None:
            rmpos = SimplifiedMPO(
                SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        else:
            ao_ops = VectorOpElement([None] * self.n_sites)
            _print('mo_coeff = ', mo_coeff)
            for ix in range(self.n_sites):
                ao_ops[ix] = ops[ix] * mo_coeff[ix]
                _print('opsx = ', ops[ix], type(ops[ix]), ao_ops[ix], type(ao_ops[ix]))
            rmpos = SimplifiedMPO(
                LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
            
        if self.mpi is not None:
            rmpos = ParallelMPO(rmpos, self.siterule)

                                
        if self.mpi is not None:
            self.mpi.barrier()
        if self.verbose >= 2:
            _print('>>> START : Applying the annihilation operator <<<')
        t = time.perf_counter()


        #==== Instantiate and setup the output MPS, rkets ====#
        if mo_coeff is None:
            rket_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target + ops.q_label, self.hamil.basis)
        else:
            rket_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target + ops[0].q_label, self.hamil.basis)
            # define new target witht the correct ionic wave function symmetry (in pyscf convention). GS_IRREP XOR removed orbital irrep.

            for i in range(self.n_sites):
                _print('qlabel : ', i, self.target, ops[i].q_label,
                       self.target + ops[i].q_label)

        
        if mo_coeff is None:
            rket_info.tag = 'DKET_%d' % idx
        else:
            rket_info.tag = 'DKET_C'
        rket_info.set_bond_dimension(mps.info.bond_dim)
        if occs is None:
            if self.verbose >= 2:
                _print("Using FCI INIT MPS")
            rket_info.set_bond_dimension(mps.info.bond_dim)
        else:
            if self.verbose >= 2:
                _print("Using occupation number INIT MPS")
            rket_info.set_bond_dimension_using_occ(
                mps.info.bond_dim, VectorDouble(occs), bias=bias)


        rket_info.save_data(self.scratch + "/" + outmps_name)
        rkets = MPS(self.n_sites, mps.center, 2)
        rkets.initialize(rket_info)
        rkets.random_canonicalize()
        rkets.save_mutable()
        rkets.deallocate()
        rket_info.save_mutable()
        rket_info.deallocate_mutable()

        #OLD if mo_coeff is None:
        #OLD     # the mpo and gf are in the same basis
        #OLD     # the mpo is SiteMPO
        #OLD     rmpos = SimplifiedMPO(
        #OLD         SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        #OLD else:
        #OLD     # the mpo is in mo basis and gf is in ao basis
        #OLD     # the mpo is sum of SiteMPO (LocalMPO)
        #OLD     ao_ops = VectorOpElement([None] * self.n_sites)
        #OLD     _print('mo_coeff = ', mo_coeff)
        #OLD     for ix in range(self.n_sites):
        #OLD         ao_ops[ix] = ops[ix] * mo_coeff[ix]
        #OLD         _print('opsx = ', ops[ix], type(ops[ix]), ao_ops[ix], type(ao_ops[ix]))
        #OLD     rmpos = SimplifiedMPO(
        #OLD         LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        #OLD     
        #OLD if self.mpi is not None:
        #OLD     rmpos = ParallelMPO(rmpos, self.siterule)


        #==== Solve for the output MPS ====#
        MPS_fitting(rkets, mps, rmpos, fit_bond_dims, fit_n_steps, fit_noises,
                    fit_conv_tol, 'density_mat', cutoff, lmpo=mpo,
                    verbose_lvl=self.verbose-1)
        _print('Output MPS max. bond dimension = ', rkets.info.bond_dim)


        #==== Normalize the output MPS if requested====#
        if outmps_normal:
            _print('Normalizing the output MPS')
            icent = rkets.center
            #OLD if rkets.dot == 2 and rkets.center == rkets.n_sites-1:
            if rkets.dot == 2:
                if rkets.center == rkets.n_sites-2:
                    icent += 1
                elif rkets.center == 0:
                    pass
            assert rkets.tensors[icent] is not None
            
            rkets.load_tensor(icent)
            rkets.tensors[icent].normalize()
            rkets.save_tensor(icent)
            rkets.unload_tensor(icent)
            # rket_info.save_data(self.scratch + "/" + outmps_name)

            
        #==== Check the norm ====#
        idMPO_ = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
        if self.mpi is not None:
            idMPO_ = ParallelMPO(idMPO_, self.identrule)
        idN = MovingEnvironment(idMPO_, rkets, rkets, "norm")
        idN.init_environments()
        nrm = Expect(idN, rkets.info.bond_dim, rkets.info.bond_dim)
        nrm_ = nrm.solve(False)
        _print('Output MPS norm = %11.8f' % nrm_)

            
        #==== Print the energy of the output MPS ====#
        energy = calc_energy_MPS(mpo, rkets, 0)
        _print('Output MPS energy = %12.8f Hartree' % energy)
        _print('Canonical form of the annihilation output = ', rkets.canonical_form)
        dm1 = self.get_one_pdm(False, rkets)
        _print('Occupations after annihilation:')
        self.print_occupation_table(dm1, aid, mo_coeff)

        
        #==== Save the output MPS ====#
        if outmps_dir != self.scratch:
            mkDir(outmps_dir)
        rket_info.save_data(outmps_dir + "/" + outmps_name)
        _print('Saving output MPS files under ' + outmps_dir)
        saveMPStoDir(rkets, outmps_dir, self.mpi)
        if save_1pdm:
            _print('Saving 1PDM of the output MPS under ' + outmps_dir)
            np.save(outmps_dir + '/ANN_1pdm', dm1)

            
        if self.verbose >= 2:
            _print('>>> COMPLETE : Application of annihilation operator | Time = %.2f <<<' %
                   (time.perf_counter() - t))
    #################################################


    #################################################
    def save_time_info(self, save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps,
                       save_1pdm, dm):

        def save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps,
                            save_1pdm, dm):
            yn_bools = ('No','Yes')
            au2fs = 2.4188843265e-2   # a.u. of time to fs conversion factor
            with open(save_dir + '/TIME_INFO', 'w') as t_info:
                t_info.write(' Actual sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
                             (it, t, t*au2fs))
                t_info.write(' Requested sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
                             (i_sp, t_sp, t_sp*au2fs))
                t_info.write(' MPS norm square = %19.14f\n' % normsq)
                t_info.write(' Autocorrelation = (%19.14f, %19.14f)\n' % (ac.real, ac.imag))
                t_info.write(f' Is MPS saved at this time?  {yn_bools[save_mps]}\n')
                t_info.write(f' Is 1PDM saved at this time?  {yn_bools[save_1pdm]}\n')
                if save_1pdm:
                    natocc_a = eigvalsh(dm[0,:,:])
                    natocc_b = eigvalsh(dm[1,:,:])
                    t_info.write(' 1PDM info:\n')
                    t_info.write('    Trace (alpha,beta) = (%16.12f,%16.12f) \n' %
                                 ( np.trace(dm[0,:,:]).real, np.trace(dm[1,:,:]).real ))
                    t_info.write('    ')
                    for i in range(0, 4+4*20): t_info.write('-')
                    t_info.write('\n')
                    t_info.write('    ' +
                                 '%4s'  % 'No.' + 
                                 '%20s' % 'Alpha MO occ.' +
                                 '%20s' % 'Beta MO occ.' +
                                 '%20s' % 'Alpha natorb occ.' +
                                 '%20s' % 'Beta natorb occ.' + '\n')
                    t_info.write('    ')
                    for i in range(0, 4+4*20): t_info.write('-')
                    t_info.write('\n')
                    for i in range(0, dm.shape[1]):
                        t_info.write('    ' +
                                     '%4d'  % i + 
                                     '%20.12f' % np.diag(dm[0,:,:])[i].real +
                                     '%20.12f' % np.diag(dm[1,:,:])[i].real +
                                     '%20.12f' % natocc_a[i].real +
                                     '%20.12f' % natocc_b[i].real + '\n')
                
        if self.mpi is not None:
            if self.mpi.rank == 0:
                save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps, save_1pdm,
                                dm)
            self.mpi.barrier()
        else:
            save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps, save_1pdm, dm)
    #################################################


    ##################################################################
    def time_propagate(self, max_bond_dim: int, method, tmax: float, dt0: float, 
                       inmps_dir0=None, inmps_name='ANN_KET', exp_tol=1e-6, cutoff=0, 
                       normalize=False, n_sub_sweeps=2, n_sub_sweeps_init=4, krylov_size=20, 
                       krylov_tol=5.0E-6, t_sample=None, save_mps=False, save_1pdm=False, 
                       save_2pdm=False, sample_dir='samples', prefix='te', prefit=False, 
                       prefit_bond_dims=None, prefit_nsteps=None, prefit_noises=None,
                       prefit_conv_tol=None, prefit_cutoff=None, verbosity=6):
        '''
        Coming soon
        '''

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        if inmps_dir0 is None:
            inmps_dir = self.scratch
        else:
            inmps_dir = inmps_dir0
        acorrfile = './' + prefix + '.ac'
        acorr2tfile = './' + prefix + '.ac2t'
        hline = ''.join(['-' for i in range(0, 61)])
        with open(acorrfile, 'w') as acf:
            acf.write('#' + hline + '\n')
            acf.write('#%9s %13s   %11s %11s %11s\n' %
                      ('No.', 'Time (a.u.)', 'Real part', 'Imag. part', 'Abs'))
            acf.write('#' + hline + '\n')
        with open(acorr2tfile, 'w') as ac2tf:
            ac2tf.write('#' + hline + '\n')
            ac2tf.write('#%9s %13s   %11s %11s %11s\n' %
                        ('No.', 'Time (a.u.)', 'Real part', 'Imag. part', 'Abs'))
            ac2tf.write('#' + hline + '\n')

                
        #==== Identity operator ====#
        idMPO = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
        print_MPO_bond_dims(idMPO, 'Identity_2')
        if self.mpi is not None:
            idMPO = ParallelMPO(idMPO, self.identrule)

            
        #==== Prepare Hamiltonian MPO ====#
        if self.mpi is not None:
            self.mpi.barrier()
        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo
        mpo = 1.0 * self.mpo_orig
        print_MPO_bond_dims(mpo, 'Hamiltonian')
        #need? mpo = IdentityAddedMPO(mpo) # hrl: alternative
        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)


        #==== Load the initial MPS ====#
        inmps_path = inmps_dir + "/" + inmps_name
        _print('Loading initial MPS info from ' + inmps_path)
        mps_info = MPSInfo(0)
        mps_info.load_data(inmps_path)
        #OLD mps = MPS(mps_info)       # This MPS-loading way does not allow loading from directories other than scratch.
        #OLD mps.load_data()           # This MPS-loading way does not allow loading from directories other than scratch.
        #OLD mps.info.load_mutable()   # This MPS-loading way does not allow loading from directories other than scratch.
        mps = loadMPSfromDir(mps_info, inmps_dir, self.mpi)

        idN = MovingEnvironment(idMPO, mps, mps, "norm_in")
        idN.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
        nrm = Expect(idN, mps.info.bond_dim, mps.info.bond_dim)
        nrm_ = nrm.solve(False)
        _print(f'Initial MPS norm = {nrm_:11.8f}')
               
        
        #==== If a change of bond dimension of the initial MPS is requested ====#
        if prefit:
            if self.mpi is not None: self.mpi.barrier()
            ref_mps = mps.deep_copy('ref_mps_t0')
            if self.mpi is not None: self.mpi.barrier()
            MPS_fitting(mps, ref_mps, idMPO, prefit_bond_dims, prefit_nsteps, prefit_noises,
                        prefit_conv_tol, 'svd', prefit_cutoff, lmpo=idMPO, verbose_lvl=self.verbose-1)


        #==== Make the initial MPS complex ====#
        cmps = MultiMPS.make_complex(mps, "mps_t")
        cmps_t0 = MultiMPS.make_complex(mps, "mps_t0")
        if mps.dot != 1: # change to 2dot      #NOTE: Is it for converting to two-site DMRG?
            cmps.load_data()
            cmps_t0.load_data()
            cmps.canonical_form = 'M' + cmps.canonical_form[1:]
            cmps_t0.canonical_form = 'M' + cmps_t0.canonical_form[1:]
            cmps.dot = 2
            cmps_t0.dot = 2
            cmps.save_data()
            cmps_t0.save_data()


        #==== Initial setups for autocorrelation ====#
        idME = MovingEnvironment(idMPO, cmps_t0, cmps, "acorr")


        #==== 1PDM IMAM
        ## pmpo = PDM1MPOQC(self.hamil)
        ## pmpo = SimplifiedMPO(pmpo, RuleQC())
        ## if self.mpi is not None:
        ##     pmpo = ParallelMPO(pmpo, self.pdmrule)
        ## pme = MovingEnvironment(pmpo, cmps, cmps, "1PDM")
            
        
        #==== Initial setups for time evolution ====#
        me = MovingEnvironment(mpo, cmps, cmps, "TE")
        self.delayed_contraction = True
        if self.delayed_contraction:
            me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments()


        #==== Time evolution ====#
        _print('Bond dim in TE : ', mps.info.bond_dim, max_bond_dim)
        if mps.info.bond_dim > max_bond_dim:
            _print('!!! WARNING !!!')
            _print('   The specified max. bond dimension for the time-evolved MPS ' +
                   f'({max_bond_dim:d}) is smaller than the max. bond dimension \n' +
                   f'  of the initial MPS ({mps.info.bond_dim:d}). This is in general not ' +
                   'recommended since the time evolution will always excite \n' +
                   '  correlation effects that are absent in the initial MPS.')
        if method == TETypes.TangentSpace:
            te = TimeEvolution(me, VectorUBond([max_bond_dim]), method)
            te.krylov_subspace_size = krylov_size
            te.krylov_conv_thrd = krylov_tol
        elif method == TETypes.RK4:
            te = TimeEvolution(me, VectorUBond([max_bond_dim]), method, n_sub_sweeps_init)
        te.cutoff = cutoff                    # for tiny systems, this is important
        te.iprint = verbosity
        te.normalize_mps = normalize
        
        #OLD n_steps = int(tmax/dt + 1)
        #OLD ts = np.linspace(0, tmax, n_steps)    # times
        if type(dt0) is not list:
            dt = [dt0]
        else:
            dt = dt0
        ts = [0.0]
        i = 1
        while ts[-1] < tmax:
            if i <= len(dt):
                ts = ts + [sum(dt[0:i])]
            else:
                ts = ts + [ts[-1] + dt[-1]]
            i += 1
        if ts[-1] > tmax:
            ts[-1] = tmax
            if abs(ts[-1]-ts[-2]) < 1E-3:
                ts.pop()
                ts[-1] = tmax
        ts = np.array(ts)
        n_steps = len(ts)
        _print('Time points (a.u.) = ', ts)
        
        
        if t_sample is not None:
            issampled = [False] * len(t_sample)
            
        i_sp = 0
        for it, tt in enumerate(ts):

            if self.verbose >= 2:
                _print('\n')
                _print(' Step : ', it)
                _print('>>> TD-PROPAGATION TIME = %10.5f <<<' %tt)
            t = time.perf_counter()

            if it != 0: # time zero: no propagation
                dt_ = ts[it] - ts[it-1]
                _print('    DELTA T = %10.5f <<<' % dt_)
                if method == TETypes.RK4:
                    te.solve(1, +1j * dt_, cmps.center == 0, tol=exp_tol)
                    te.n_sub_sweeps = n_sub_sweeps
                elif method == TETypes.TangentSpace:
                    te.solve(2, +1j * dt_ / 2, cmps.center == 0, tol=exp_tol)
                    te.n_sub_sweeps = 1                    

            #==== Autocorrelation ====#
            idME.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
            acorr = ComplexExpect(idME, max_bond_dim, max_bond_dim)
            acorr_t = acorr.solve(False)  
            _print('Autocorrelation function = ' +
                   f'{acorr_t.real:11.8f} (Re), {acorr_t.imag:11.8f} (Im), ' +
                   f'{abs(acorr_t):11.8f} (Abs)')
            with open(acorrfile, 'a') as acf:
                acf.write(' %9d %13.8f   %11.8f %11.8f %11.8f\n' %
                          (it, tt, acorr_t.real, acorr_t.imag, abs(acorr_t)) )
            if tt > tmax/2:
                if cmps.wfns[0].data.size == 0:
                    loaded = True
                    cmps.load_tensor(cmps.center)
                vec = cmps.wfns[0].data + 1j * cmps.wfns[1].data
                acorr_2t = np.vdot(vec.conj(),vec)
                with open(acorr2tfile, 'a') as ac2tf:
                    ac2tf.write(' %9d %13.8f   %11.8f %11.8f %11.8f\n' %
                                (it, 2*tt, acorr_2t.real, acorr_2t.imag, abs(acorr_2t)) )

            
            #==== 1PDM IMAM
            ## pme.init_environments()
            ## _print('here1')
            ## expect = ComplexExpect(pme, max_bond_dim+100, max_bond_dim+100)   #NOTE
            ## _print('here2')
            ## expect.solve(True, cmps.center == 0)    #NOTE: setting the 1st param to True makes cmps real.
            ## _print('here3')
            ## if SpinLabel == SU2:
            ##     dmr = expect.get_1pdm_spatial(self.n_sites)
            ##     dm = np.array(dmr).copy()
            ## else:
            ##     dmr = expect.get_1pdm(self.n_sites)
            ##     dm = np.array(dmr).copy()
            ##     dm = dm.reshape((self.n_sites, 2, self.n_sites, 2))
            ##     dm = np.transpose(dm, (0, 2, 1, 3))
            ## _print('here4')
            ## cmps.save_data()
            ## if self.ridx is not None:
            ##     dm[:, :] = dm[self.ridx, :][:, self.ridx]
            ## _print('here5')
            ## dmr.deallocate()
            ## _print('here6')
            ## if SpinLabel == SU2:
            ##     dm = np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
            ## else:
            ##     dm = np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
            ## _print('here7')
            ## _print('splabel ', SpinLabel)
            ## _print('DM eigvals = ', dm.shape, scipy.linalg.norm(dm), dm.dtype)


            #==== Stores MPS and/or PDM's at sampling times ====#
            if t_sample is not None and np.prod(issampled)==0:
                if it < n_steps-1:
                    dt1 = abs( ts[it]   - t_sample[i_sp] )
                    dt2 = abs( ts[it+1] - t_sample[i_sp] )
                    dd = dt1 < dt2
                else:
                    dd = True
                    
                if dd and not issampled[i_sp]:
                    save_dir = sample_dir + '/mps_sp-' + str(i_sp)
                    if self.mpi is not None:
                        if self.mpi.rank == 0:
                            mkDir(save_dir)
                            self.mpi.barrier()
                    else:
                        mkDir(save_dir)

            
                    #==== Saving MPS ====##
                    if save_mps:
                        saveMPStoDir(cmps, save_dir, self.mpi)

                    #==== Saving 1PDM ====#
                    dm = None
                    if save_1pdm:
                        #== Copy the current MPS because self.get_one_pdm ==#
                        #==      convert the input MPS to a real MPS      ==#
                        if self.mpi is not None: self.mpi.barrier()
                        cmps_cp = cmps.deep_copy('cmps_cp')
                        if self.mpi is not None: self.mpi.barrier()
                        
                        dm = self.get_one_pdm(True, cmps_cp)
                        np.save(save_dir+'/1pdm', dm)
                        cmps_cp.info.deallocate()
                        ## cmps_cp.deallocate()      # Unnecessary because it must have already been called inside the expect.solve function in the get_one_pdm above


                    #==== Save time info ====#
                    if it == 0:
                        normsqs = abs(acorr_t)
                    elif it > 0:
                        normsqs = te.normsqs[0]
                    self.save_time_info(save_dir, ts[it], it, t_sample[i_sp], i_sp, 
                                        normsqs, acorr_t, save_mps, save_1pdm, dm)
                    
                    issampled[i_sp] = True
                    i_sp += 1
    ##############################################################
    

    ##############################################################
    def __del__(self):
        if self.hamil is not None:
            self.hamil.deallocate()
        if self.fcidump is not None:
            self.fcidump.deallocate()
        if self.mpo_orig is not None:
            self.mpo_orig.deallocate()
        release_memory()

##############################################################





# A section from gfdmrg.py about the definition of the dmrg_mo_gf function has been
# removed.


# A section from gfdmrg.py has been removed.







#In ft_tddmrg.py, what does MYTDDMRG.fmt_size mean? This quantity exist inside the MYTDDMRG class
#definition. Shouldn't it be self.fmt_size.
#It looks like the RT_MYTDDMRG inherits from FTDMRG class, but is there no super().__init__ method in
#its definition?
#What does these various deallocate actually do? Somewhere an mps is deallocated (by calling mps.deallocate())
#but then the same mps is still used to do something.
#ANS: ME writes/reads mps to/from disk that's why mps can be deallocated although it is used again later.
