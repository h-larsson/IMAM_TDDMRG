
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
from scipy.linalg import eigvalsh, eigh

# Set spin-adapted or non-spin-adapted here
SpinLabel = SU2
#SpinLabel = SZ

if SpinLabel == SU2:
    from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, IdentityMPO
    from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
    from block2.su2 import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
    try:
        from block2.su2 import MPICommunicator
        hasMPI = True
    except ImportError:
        hasMPI = False
else:
    from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
    from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, IdentityMPO
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
from IMAM_TDDMRG.utils.util_print import getVerbosePrinter, print_section, print_describe_content
from IMAM_TDDMRG.utils.util_print import print_orb_occupations, print_pcharge, print_mpole
from IMAM_TDDMRG.utils.util_print import print_td_pcharge, print_td_mpole
from IMAM_TDDMRG.utils.util_qm import make_full_dm
from IMAM_TDDMRG.utils.util_mps import print_MPO_bond_dims, MPS_fitting, calc_energy_MPS
from IMAM_TDDMRG.observables import pcharge, mpole
from IMAM_TDDMRG.phys_const import au2fs

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
class MYTDDMRG:
    """
    DDMRG++ for Green's Function for molecules.
    """


    #################################################
    def __init__(self, mol, nel_site, scratch='./nodex', memory=1*1E9, isize=2E8, 
                 omp_threads=8, verbose=2, print_statistics=True, mpi=None,
                 delayed_contraction=True):
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

        #==== Some persistent quantities ====#
        self.mol = mol
        assert isinstance(nel_site, tuple), 'init: The argument nel_site must be a tuple.'
        self.nel = sum(mol.nelec)
        self.nel_site = nel_site
        self.nel_core = self.nel - sum(nel_site)
        assert self.nel_core%2 == 0, \
            f'The number of core electrons (currently {self.nel_core}) must be an even ' + \
            'number.'
        self.ovl_ao = mol.intor('int1e_ovlp')
        mol.set_common_origin([0,0,0])
        self.dpole_ao = mol.intor('int1e_r').reshape(3,mol.nao,mol.nao)
        self.qpole_ao = mol.intor('int1e_rr').reshape(3,3,mol.nao,mol.nao)
            
    #################################################


    #################################################
    def assign_orbs(self, n_core, n_sites, orbs):

        if SpinLabel == SU2:
            n_mo = orbs.shape[1]
            assert orbs.shape[0] == orbs.shape[1]
        elif SpinLabel == SZ:
            n_mo = orbs.shape[2]
            assert orbs.shape[1] == orbs.shape[2]
        assert n_mo == self.mol.nao
        n_occ = n_core + n_sites
        n_virt = n_mo - n_occ

        if SpinLabel == SU2:
            assert len(orbs.shape) == 2, \
                'If SU2 symmetry is invoked, orbs must be a 2D array.'
            orbs_c = np.zeros((2, n_mo, n_core))
            orbs_s = np.zeros((2, n_mo, n_sites))
            orbs_v = np.zeros((2, n_mo, n_virt))
            for i in range(0,2):
                orbs_c[i,:,:] = orbs[:, 0:n_core]
                orbs_s[i,:,:] = orbs[:, n_core:n_occ]
                orbs_v[i,:,:] = orbs[:, n_occ:n_mo]
        elif SpinLabel == SZ:
            assert len(orbs.shape) == 3 and orbs.shape[0] == 2, \
                'If SZ symmetry is invoked, orbs must be a 3D array with the size ' + \
                'of first dimension being two.'
            orbs_c = orbs[:, :, 0:n_core]
            orbs_s = orbs[:, :, n_core:n_occ]
            orbs_v = orbs[:, :, n_occ:n_mo]

        return orbs_c, orbs_s, orbs_v
    #################################################

            
    #################################################
    def init_hamiltonian_fcidump(self, pg, filename, orbs, idx=None):
        """Read integrals from FCIDUMP file."""
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        self.groupname = pg
        assert self.fcidump.n_elec == sum(self.nel_site), \
            f'init_hamiltonian_fcidump: self.fcidump.n_elec ({self.fcidump.n_elec}) must ' + \
            'be identical to sum(self.nel_site) (%d).' % sum(self.nel_site)

        #==== Reordering indices ====#
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)

        #==== Orbitals and MPS symemtries ====#
        swap_pg = getattr(PointGroup, "swap_" + pg)
        self.orb_sym = VectorUInt8(map(swap_pg, self.fcidump.orb_sym))      # 1)
        _print("# fcidump symmetrize error:", self.fcidump.symmetrize(orb_sym))
        self.wfn_sym = swap_pg(self.fcidump.isym)
        # NOTE:
        # 1) Because of the self.fcidump.reorder invocation above, self.orb_sym contains
        #    orbital symmetries AFTER REORDERING.

        #==== Construct the Hamiltonian MPO ====#
        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                swap_pg(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites
        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        #==== Assign orbitals ====#
        self.core_orbs, self.site_orbs, self.virt_orbs = \
            self.assign_orbs(int(self.nel_core/2), self.n_sites, orbs)
        self.n_core, self.n_virt = self.core_orbs.shape[2], self.virt_orbs.shape[2]
        self.n_orbs = self.n_core + self.n_sites + self.n_virt

        #==== Reordering orbitals ====#
        if idx is not None:
            self.site_orbs = self.site_orbs[:,:,idx]
    #################################################


    #################################################
    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym, e_core, 
                         h1e, g2e, orbs, tol=1E-13, idx=None, save_fcidump=None):
        """
        Initialize integrals using h1e, g2e, etc.
        n_elec : The number of electrons within the sites. This means, if there are core
                 orbitals, then n_elec are the number of electrons in the active space only.
        isym: wfn symmetry in molpro convention? See the getFCIDUMP function in CAS_example.py.
        g2e: Does it need to be in 8-fold symmetry? See the getFCIDUMP function in CAS_example.py.
        orb_sym: orbitals symmetry in molpro convention.
        """

        #==== Initialize self.fcidump ====#
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.groupname = pg
        assert n_elec == sum(self.nel_site), \
            f'init_hamiltonian: The argument n_elec ({n_elec}) must be identical to ' + \
            'sum(self.nel_site) (%d).' % sum(self.nel_site)

        #==== Rearrange the 1e and 2e integrals, and initialize FCIDUMP ====#
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
            assert twos == 2*(self.nel_site[0]-self.nel_site[1]), \
                'init_hamiltonian: When SZ symmetry is enabled, the argument twos must be ' + \
                'equal to twice the difference between alpha and beta electrons. ' + \
                f'Currently, their values are twos = {twos} and 2*(n_alpha - n_beta) = ' + \
                f'{2*(self.nel_site[0]-self.nel_site[1])}.'
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
            #OLD mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            mg2e = tuple(xg2e.ravel() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0      # xmg2e works like a pointer to the elements of mg2e tuple.
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)

        #==== Take care of the symmetry conventions. Note that ====#
        #====   self.fcidump.orb_sym is in Molpro convention,  ====#
        #====     while self.orb_sym is in block2 convention   ====#
        self.fcidump.orb_sym = VectorUInt8(orb_sym)       # Hence, self.fcidump.orb_sym is in Molpro convention.

        #==== Reordering indices ====#
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)

        #==== Orbitals and MPS symemtries ====#
        swap_pg = getattr(PointGroup, "swap_" + pg)
        self.orb_sym = VectorUInt8(map(swap_pg, self.fcidump.orb_sym))      # 1)
        self.wfn_sym = swap_pg(isym)
        # NOTE:
        # 1) Because of the self.fcidump.reorder invocation above, self.orb_sym contains
        #    orbital symmetries AFTER REORDERING.


        #==== Construct the Hamiltonian MPO ====#
        vacuum = SpinLabel(0)
        self.target = SpinLabel(n_elec, twos, swap_pg(isym))
        self.n_sites = n_sites
        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        #==== Assign orbitals ====#
        self.core_orbs, self.site_orbs, self.virt_orbs = \
            self.assign_orbs(int(self.nel_core/2), self.n_sites, orbs)
        self.n_core, self.n_virt = self.core_orbs.shape[2], self.virt_orbs.shape[2]
        self.n_orbs = self.n_core + self.n_sites + self.n_virt

        #==== Reorder orbitals ====#
        if idx is not None:
            self.site_orbs = self.site_orbs[:,:,idx]
        
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
    def unordered_site_orbs(self):
        
        if self.ridx is None:
            return self.site_orbs
        else:
            return self.site_orbs[:,:,self.ridx]
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
                #ERR occs = self.fcidump.reorder(VectorDouble(occs), VectorUInt16(self.idx))
                occs = occs[self.idx]
            mps_info.set_bond_dimension_using_occ(
                bond_dims[0], VectorDouble(occs), bias=bias)
        
        mps = MPS(self.n_sites, 0, 2)   # The 3rd argument controls the use of one/two-site algorithm.
        mps.initialize(mps_info)
        mps.random_canonicalize()

        mps.save_mutable()
        mps.deallocate()
        mps_info.save_mutable()
        mps_info.deallocate_mutable()
        

        # MPO
        tx = time.perf_counter()
        mpo = MPOQC(self.hamil, QCTypes.Conventional)
        mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                            OpNamesSet((OpNames.R, OpNames.RD)))
        self.mpo_orig = mpo

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
        _print("Ground state energy = %20.15f" % self.gs_energy)


        #==== MO occupations ====#
        dm0 = self.get_one_pdm(False, mps)
        dm0_full = make_full_dm(self.n_core, dm0)
        occs0 = np.zeros((2, self.n_core+self.n_sites))
        for i in range(0, 2): occs0[i,:] = np.diag(dm0_full[i,:,:]).copy()
        print_orb_occupations(occs0)
        
            
        #==== Partial charge ====#
        orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()), axis=2)
        self.qmul0, self.qlow0 = \
            pcharge.calc(self.mol, dm0_full, orbs, self.ovl_ao)
        print_pcharge(self.mol, self.qmul0, self.qlow0)


        #==== Multipole analysis ====#
        e_dpole, n_dpole, e_qpole, n_qpole = \
            mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm0_full, orbs)
        print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)

        
        #==== Save the output MPS ====#
        #OLD mps.save_data()
        #OLD mps_info.save_data(self.scratch + "/GS_MPS_INFO")
        #OLD mps_info.deallocate()
        _print('')
        _print('Saving the ground state MPS files under ' + outmps_dir)
        if outmps_dir != self.scratch:
            mkDir(outmps_dir)
        mps_info.save_data(outmps_dir + "/" + outmps_name)
        saveMPStoDir(mps, outmps_dir, self.mpi)
        _print('Output ground state max. bond dimension = ', mps.info.bond_dim)
        if save_1pdm:
            np.save(outmps_dir + '/GS_1pdm', dm0)


        #==== Statistics ====#
        if self.print_statistics:
            dmain, dseco, imain, iseco = Global.frame.peak_used_memory
            _print("GS PEAK MEM USAGE:",
                   "DMEM = ", MYTDDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", MYTDDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))


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
    def print_occupation_table(self, dm, aorb):

        from pyscf import symm
        
        natocc_a = eigvalsh(dm[0,:,:])
        natocc_a = natocc_a[::-1]
        natocc_b = eigvalsh(dm[1,:,:])
        natocc_b = natocc_b[::-1]

        mk = ''
        if isinstance(aorb, int):
            mk = ' (ann.)'
        ll = 4 + 16 + 15 + 12 + (0 if isinstance(aorb, int) else 13) + \
             (3+18) + 18 + 2*len(mk)
        hline = ''.join(['-' for i in range(0, ll)])
        aspace = ''.join([' ' for i in range(0,len(mk))])

        _print(hline)
        _print('%4s'  % 'No.', end='') 
        _print('%16s' % 'Alpha MO occ.' + aspace, end='')
        _print('%15s' % 'Beta MO occ.' + aspace,  end='')
        _print('%13s' % 'Irrep / ID',  end='')
        if isinstance(aorb, np.ndarray):
            _print('%13s' % 'aorb coeff', end='')
        _print('   %18s' % 'Alpha natorb occ.', end='')
        _print('%18s' % 'Beta natorb occ.',  end='\n')
        _print(hline)

        for i in range(0, dm.shape[1]):
            if isinstance(aorb, int):
                mk0 = aspace
                if i == aorb: mk0 = mk
            else:
                mk0 = aspace

            _print('%4d' % i, end='')
            _print('%16.8f' % np.diag(dm[0,:,:])[i] + mk0, end='')
            _print('%15.8f' % np.diag(dm[1,:,:])[i] + mk0, end='')
            j = i if self.ridx is None else self.ridx[i]
            sym_label = symm.irrep_id2name(self.groupname, self.orb_sym[j])
            _print('%13s' % (sym_label + ' / ' + str(self.orb_sym[j])), end='')
            if isinstance(aorb, np.ndarray):
                _print('%13.8f' % aorb[i], end='')
            _print('   %18.8f' % natocc_a[i], end='')
            _print('%18.8f' % natocc_b[i], end='\n')

        _print(hline)
        _print('%4s' % 'Sum', end='')
        _print('%16.8f' % np.trace(dm[0,:,:]) + aspace, end='')
        _print('%15.8f' % np.trace(dm[1,:,:]) + aspace, end='')
        _print('%13s' % ' ', end='')
        if isinstance(aorb, np.ndarray):
            _print('%13s' % ' ', end='')
        _print('   %18.8f' % sum(natocc_a), end='')
        _print('%18.8f' % sum(natocc_b), end='\n')
        _print(hline)
    #################################################

    
    #################################################
    def annihilate(self, aorb, fit_bond_dims, fit_noises, fit_conv_tol, fit_n_steps, pg,
                   inmps_dir0=None, inmps_name='GS_MPS_INFO', outmps_dir0=None,
                   outmps_name='ANN_KET', aorb_thr=1.0E-12, alpha=True, 
                   cutoff=1E-14, occs=None, bias=1.0, outmps_normal=True, save_1pdm=False):
        """
        aorb can be int, numpy.ndarray, or 'nat<n>' where n is an integer'
        """
        ##OLD ops = [None] * len(aorb)
        ##OLD rkets = [None] * len(aorb)
        ##OLD rmpos = [None] * len(aorb)
        from pyscf import symm

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

            
        #==== Checking input parameters ====#
        if not (isinstance(aorb, int) or isinstance(aorb, np.ndarray) or
                isinstance(aorb, str)):
            raise ValueError('The argument \'aorb\' of MYTDDMRG.annihilate method must ' +
                             'be either an integer, a numpy.ndarray, or a string. ' +
                             f'Currently, aorb = {aorb}.')

        use_natorb = False
        if isinstance(aorb, str):
            nn = len(aorb)
            ss = aorb[0:3]
            ss_ = aorb[3:nn]
            if ss != 'nat' or not ss_.isdigit:
                _print('aorb = ', aorb)
                raise ValueError(
                    'The only way the argument \'aorb\' of MYTDDMRG.annihilate ' +
                    'can be a string is when it has the format of \'nat<n>\', ' +
                    'where \'n\' must be an integer, e.g. \'nat1\', \'nat24\', ' +
                    f'etc. Currently aorb = {aorb:s}')
            nat_id = int(ss_)
            if nat_id < 0 or nat_id >= self.n_sites:
                raise ValueError('The index of the natural orbital specified by ' +
                                 'the argument \'aorb\' of MYTDDMRG.annihilate ' +
                                 'is out of bound, which is between 0 and ' +
                                 f'{self.n_sites:d}. Currently, the specified index ' +
                                 f'is {nat_id:d}.')
            use_natorb = True

            
        idMPO_ = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
        if self.mpi is not None:
            idMPO_ = ParallelMPO(idMPO_, self.identrule)

        

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


        #==== Compute the requested natural orbital ====#
        dm0 = self.get_one_pdm(False, mps)
        if use_natorb:
            e0, aorb = eigh(dm0[0 if alpha else 1,:,:])
            aorb = aorb[:,::-1]       # Reverse the columns
            aorb = aorb[:,nat_id]
            aorb[np.abs(aorb) < aorb_thr] = 0.0          # 1)
            _print(aorb)
        _print('Occupations before annihilation:')
        self.print_occupation_table(dm0, aorb)
        # NOTES:
        # 1) For some reason, without setting the small coefficients to zero,
        #    a segfault error happens later inside the MPS_fitting function.

        
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

        
        #==== Determine the reordered index of the annihilated orbital ====#
        if isinstance(aorb, int):
            idx = aorb if self.ridx is None else self.ridx[aorb]
            # idx is the index of the annihilated orbital after reordering.
        elif isinstance(aorb, np.ndarray):
            if self.idx is not None:
                aorb = aorb[self.idx]            
            
            #== Determine the irrep. of aorb ==#
            for i in range(0, self.n_sites):
                j = i if self.ridx is None else self.ridx[i]
                if (np.abs(aorb[j]) >= aorb_thr):
                    aorb_sym = self.orb_sym[j]            # 1)
                    break
            for i in range(0, self.n_sites):
                j = i if self.ridx is None else self.ridx[i]
                if (np.abs(aorb[j]) >= aorb_thr and self.orb_sym[j] != aorb_sym):
                    _print(self.orb_sym)
                    _print('An inconsistency in the orbital symmetry found in aorb: ')
                    _print(f'   The first detected nonzero element has a symmetry ID of {aorb_sym:d}, ', end='')
                    _print(f'but the symmetry ID of another nonzero element (the {i:d}-th element) ', end='')
                    _print(f'is {self.orb_sym[j]:d}.')
                    raise ValueError('The orbitals making the linear combination in ' +
                                     'aorb must all have the same symmetry.')
            # NOTES:
            # 1) j instead of i is used as the index for self.orb_sym because the
            #    contents of this array have been reordered (see its assignment in the
            #    self.init_hamiltonian or self.init_hamiltonian_fcidump function).

                
        #==== Begin constructing the annihilation MPO ====#
        gidxs = list(range(self.n_sites))
        if isinstance(aorb, int):
            if SpinLabel == SZ:
                ops = OpElement(OpNames.D, SiteIndex((idx, ), (0 if alpha else 1, )), 
                    SZ(-1, -1 if alpha else 1, self.orb_sym[idx]))
            else:
                ops = OpElement(OpNames.D, SiteIndex((idx, ), ()), 
                    SU2(-1, 1, self.orb_sym[idx]))
        elif isinstance(aorb, np.ndarray):
            ops = [None] * self.n_sites
            for ii, ix in enumerate(gidxs):
                if SpinLabel == SZ:
                    ops[ii] = OpElement(OpNames.D, SiteIndex((ix, ), (0 if alpha else 1, )),
                        SZ(-1, -1 if alpha else 1, aorb_sym))
                else:
                    ops[ii] = OpElement(OpNames.D, SiteIndex((ix, ), ()), 
                        SU2(-1, 1, aorb_sym))
                    

        #==== Determine if the annihilated orbital is a site orbital ====#
        #====  or a linear combination of them (orbital transform)   ====#
        if isinstance(aorb, int):
            rmpos = SimplifiedMPO(
                SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        elif isinstance(aorb, np.ndarray):
            ao_ops = VectorOpElement([None] * self.n_sites)
            for ix in range(self.n_sites):
                ao_ops[ix] = ops[ix] * aorb[ix]
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


        #==== Determine the quantum numbers of the output MPS, rkets ====#
        if isinstance(aorb, int):
            ion_target = self.target + ops.q_label
        elif isinstance(aorb, np.ndarray):
            ion_sym = self.wfn_sym ^ aorb_sym
            ion_target = SpinLabel(sum(self.nel_site)-1, 1, ion_sym)
        rket_info = MPSInfo(self.n_sites, self.hamil.vacuum, ion_target, self.hamil.basis)
        
        _print('Quantum number information:')
        _print(' - Input MPS = ', self.target)
        _print(' - Input MPS multiplicity = ', self.target.multiplicity)
        if isinstance(aorb, int):
            _print(' - Annihilated orbital = ', ops.q_label)
        elif isinstance(aorb, np.ndarray):
            _print(' - Annihilated orbital = ', SpinLabel(-1, 1, aorb_sym))
        _print(' - Output MPS = ', ion_target)
        _print(' - Output MPS multiplicity = ', ion_target.multiplicity)


        #==== Tag the output MPS ====#
        if isinstance(aorb, int):
            rket_info.tag = 'DKET_%d' % idx
        elif isinstance(aorb, np.ndarray):
            rket_info.tag = 'DKET_C'
        

        #==== Set the bond dimension of output MPS ====#
        rket_info.set_bond_dimension(mps.info.bond_dim)
        if occs is None:
            if self.verbose >= 2:
                _print("Using FCI INIT MPS")
            rket_info.set_bond_dimension(mps.info.bond_dim)
        else:
            if self.verbose >= 2:
                _print("Using occupation number INIT MPS")
            if self.idx is not None:
                occs = occs[self.idx]
            rket_info.set_bond_dimension_using_occ(
                mps.info.bond_dim, VectorDouble(occs), bias=bias)


        #==== Initialization of output MPS ====#
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


        #==== Normalize the output MPS if requested ====#
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
        if isinstance(aorb, int):
            self.print_occupation_table(dm1, aorb)
        elif isinstance(aorb, np.ndarray):
            if self.ridx is None:
                self.print_occupation_table(dm1, aorb)
            else:
                self.print_occupation_table(dm1, aorb[self.ridx])

            
        #==== Partial charge ====#
        dm1_full = make_full_dm(self.n_core, dm1)
        orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()), axis=2)
        self.qmul1, self.qlow1 = \
            pcharge.calc(self.mol, dm1_full, orbs, self.ovl_ao)
        print_pcharge(self.mol, self.qmul1, self.qlow1)


        #==== Multipole analysis ====#
        e_dpole, n_dpole, e_qpole, n_qpole = \
            mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm1_full, orbs)
        print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)

        
        #==== Save the output MPS ====#
        _print('')
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
                       save_2pdm=False, sample_dir='samples', prefix='te', save_txt=True,
                       save_npy=False, prefit=False, prefit_bond_dims=None, 
                       prefit_nsteps=None, prefit_noises=None, prefit_conv_tol=None, 
                       prefit_cutoff=None, verbosity=6):
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

        #==== Initiate autocorrelation file ====#
        hline = ''.join(['-' for i in range(0, 73)])
        ac_file = './' + prefix + '.ac'
        if self.mpi is None or self.mpi.rank == 0:
            with open(ac_file, 'w') as acf:
                print_describe_content('autocerrelation data', acf)
                acf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
                acf.write('#' + hline + '\n')
                acf.write('#%9s %13s   %11s %11s %11s %11s\n' %
                          ('No.', 'Time (a.u.)', 'Real part', 'Imag. part', 'Abs', 'Norm'))
                acf.write('#' + hline + '\n')

        #==== Initiate 2t autocorrelation file ====#
        ac2t_f = './' + prefix + '.ac2t'
        if self.mpi is None or self.mpi.rank == 0:
            with open(ac2t_f, 'w') as ac2tf:
                print_describe_content('2t-autocerrelation data', ac2tf)
                ac2tf.write('# 1 a.u. of time = %.10f fs\n' % au2fs)
                ac2tf.write('#' + hline + '\n')
                ac2tf.write('#%9s %13s   %11s %11s %11s\n' %
                            ('No.', 'Time (a.u.)', 'Real part', 'Imag. part', 'Abs'))
                ac2tf.write('#' + hline + '\n')

        #==== Initiate Lowdin partial charges file ====#
        if t_sample is not None:
            atom_symbol = [self.mol.atom_symbol(i) for i in range(0, self.mol.natm)]
            q_print = print_td_pcharge(atom_symbol, prefix, len(t_sample), 8, save_txt,
                                       save_npy)
            if self.mpi is None or self.mpi.rank == 0: q_print.header()
            if self.mpi is not None: self.mpi.barrier()

        #==== Initiate multipole components file ====#
        if t_sample is not None:
            mp_print = print_td_mpole(prefix, len(t_sample), save_txt, save_npy)
            if self.mpi is None or self.mpi.rank == 0: mp_print.header()
            if self.mpi is not None: self.mpi.barrier()
                
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
        nel_t0 = mps_info.target.n + self.nel_core
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
                        prefit_conv_tol, 'density_mat', prefit_cutoff, lmpo=idMPO,
                        verbose_lvl=self.verbose-1)


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
        
        
        #==== Construct the time vector ====#
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


        #==== Begin the time evolution ====#
        acorr_t = np.zeros((len(ts)), dtype=np.complex128)
        acorr_2t = np.zeros((len(ts)), dtype=np.complex128)
        if save_npy:
            np.save('./'+prefix+'.t', ts)
            if t_sample is not None: np.save('./'+prefix+'.ts', t_sample)
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

            
            #==== Autocorrelation and norm ====#
            idME.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
            acorr = ComplexExpect(idME, max_bond_dim, max_bond_dim)
            acorr_t[it] = acorr.solve(False)
            if it == 0:
                normsqs = abs(acorr_t[it])
            elif it > 0:
                normsqs = te.normsqs[0]
            acorr_t[it] = acorr_t[it] / np.sqrt(normsqs)
                
            #==== Print autocorrelation ====#
            if save_txt and (self.mpi is None or self.mpi.rank == 0):
                with open(ac_file, 'a') as acf:
                    acf.write(' %9d %13.8f   %11.8f %11.8f %11.8f %11.8f\n' %
                              (it, tt, acorr_t[it].real, acorr_t[it].imag, abs(acorr_t[it]),
                               normsqs) )
            if save_npy and (self.mpi is None or self.mpi.rank == 0):
                np.save(ac_file, acorr_t[0:it+1])

            #==== 2t autocorrelation ====#
            if cmps.wfns[0].data.size == 0:
                loaded = True
                cmps.load_tensor(cmps.center)
            vec = cmps.wfns[0].data + 1j * cmps.wfns[1].data
            acorr_2t[it] = np.vdot(vec.conj(),vec) / normsqs

            #==== Print 2t autocorrelation ====#
            if save_txt and (self.mpi is None or self.mpi.rank == 0):
                with open(ac2t_f, 'a') as ac2tf:
                    ac2tf.write(' %9d %13.8f   %11.8f %11.8f %11.8f\n' %
                                (it, 2*tt, acorr_2t[it].real, acorr_2t[it].imag,
                                 abs(acorr_2t[it])) )
            if save_npy and (self.mpi is None or self.mpi.rank == 0):
                np.save(ac2t_f, acorr_2t[0:it+1])
            if self.mpi is not None: self.mpi.barrier()

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

                    #==== Calculate 1PDM ====#
                    if self.mpi is not None: self.mpi.barrier()
                    cmps_cp = cmps.deep_copy('cmps_cp')         # 1)
                    if self.mpi is not None: self.mpi.barrier()
                    dm = self.get_one_pdm(True, cmps_cp)
                    cmps_cp.info.deallocate()
                    dm_full = make_full_dm(self.n_core, dm)
                    dm_tr = np.sum( np.trace(dm_full, axis1=1, axis2=2) )
                    dm_full = dm_full * nel_t0 / np.abs(dm_tr)      # fm_full is now normalized
                    #OLD cmps_cp.deallocate()      # Unnecessary because it must have already been called inside the expect.solve function in the get_one_pdm above
                    # NOTE:
                    # 1) Copy the current MPS because self.get_one_pdm convert the input
                    #    MPS to a real MPS.

                    #==== Save 1PDM ====#
                    if save_1pdm: np.save(save_dir+'/1pdm', dm)
                        
                    #==== Save time info ====#
                    self.save_time_info(save_dir, ts[it], it, t_sample[i_sp], i_sp, normsqs, 
                                        acorr_t[it], save_mps, save_1pdm, dm)

                    #==== Partial charges ====#
                    orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()),
                                          axis=2)
                    qmul, qlow = pcharge.calc(self.mol, dm_full, orbs, self.ovl_ao)
                    if self.mpi is None or self.mpi.rank == 0:
                        q_print.print_pcharge(tt, qlow)
                    if self.mpi is not None: self.mpi.barrier()

                    #==== Multipole components ====#
                    e_dpole, n_dpole, e_qpole, n_qpole = \
                        mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm_full, orbs)
                    if self.mpi is None or self.mpi.rank == 0:
                        mp_print.print_mpole(tt, e_dpole, n_dpole, e_qpole, n_qpole)
                    if self.mpi is not None: self.mpi.barrier()


                    issampled[i_sp] = True
                    i_sp += 1

        #==== Print max min imaginary parts (for debugging) ====#
        if self.mpi is None or self.mpi.rank == 0:
            q_print.footer()

        if self.mpi is None or self.mpi.rank == 0:
            mp_print.footer()
            
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
