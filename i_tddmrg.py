
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
#OLD from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
#OLD from block2 import init_memory, release_memory, SiteIndex
#OLD from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
#OLD from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
#OLD from block2 import OrbitalOrdering, VectorUInt16, TETypes

import time
import numpy as np
import scipy.linalg
from scipy.linalg import eigvalsh, eigh

# Set spin-adapted or non-spin-adapted here
#OLD_CPX SpinLabel = SU2
#OLD_CPX #SpinLabel = SZ
spin_symmetry = 'su2'
#spin_symmetry = 'sz'


iscomp = False
import block2 as b2
if iscomp:
    bx = b2.cpx
    bc = bx
else:
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
    hasMPI = False



#OLD_CPX if SpinLabel == SU2:
#OLD_CPX     from block2.su2 import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
#OLD_CPX     from block2.su2 import MPSInfo, MPS, MovingEnvironment, DMRG, IdentityMPO
#OLD_CPX     from block2.su2 import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
#OLD_CPX     from block2.su2 import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
#OLD_CPX     try:
#OLD_CPX         from block2.su2 import MPICommunicator
#OLD_CPX         hasMPI = True
#OLD_CPX     except ImportError:
#OLD_CPX         hasMPI = False
#OLD_CPX else:
#OLD_CPX     from block2.sz import HamiltonianQC, SimplifiedMPO, Rule, RuleQC, MPOQC
#OLD_CPX     from block2.sz import MPSInfo, MPS, MovingEnvironment, DMRG, IdentityMPO
#OLD_CPX     from block2.sz import OpElement, SiteMPO, NoTransposeRule, PDM1MPOQC, Expect, ComplexExpect
#OLD_CPX     from block2.sz import VectorOpElement, LocalMPO, MultiMPS, TimeEvolution
#OLD_CPX     try:
#OLD_CPX         from block2.sz import MPICommunicator
#OLD_CPX         hasMPI = True
#OLD_CPX     except ImportError:
#OLD_CPX         hasMPI = False

import tools; tools.init(SX)
from tools import saveMPStoDir, loadMPSfromDir, mkDir
from gfdmrg import orbital_reorder
from IMAM_TDDMRG.utils.util_print import getVerbosePrinter, print_section, print_describe_content
from IMAM_TDDMRG.utils.util_print import print_orb_occupations, print_pcharge, print_mpole
from IMAM_TDDMRG.utils.util_print import print_autocorrelation, print_td_pcharge, print_td_mpole
from IMAM_TDDMRG.utils.util_qm import make_full_dm, get_one_pdm
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


        if spin_symmetry == 'sz':
            _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            _print('WARNING: SZ Spin label is chosen! The MYTDDMRG class was designed ' +
                   'with the SU2 spin label in mind. The use of SZ spin label in this ' +
                   'class has not been checked.')
            _print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        _print('Memory = %10.2f Megabytes' % (memory/1.0e6))
        _print('Integer size = %10.2f Megabytes' % (isize/1.0e6))

        
        b2.Random.rand_seed(0)
        isize = int(isize)
        assert isize < memory
        #OLD isize = min(int(memory * 0.1), 200000000)
        b2.init_memory(isize=isize, dsize=int(memory - isize), save_dir=scratch)
        b2.Global.threading = b2.Threading(
            b2.ThreadingTypes.OperatorBatchedGEMM | b2.ThreadingTypes.Global, omp_threads,
            omp_threads, 1)
        b2.Global.threading.seq_type = b2.SeqTypes.Tasked
        b2.Global.frame.load_buffering = False
        b2.Global.frame.save_buffering = False
        b2.Global.frame.use_main_stack = False
        b2.Global.frame.minimal_disk_usage = True

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
            _print(b2.Global.frame)
            _print(b2.Global.threading)

        if mpi is not None:
            #OLD_CPX if SpinLabel == SU2:
            #OLD_CPX     from block2.su2 import ParallelRuleQC, ParallelRuleNPDMQC, ParallelRuleSiteQC
            #OLD_CPX     from block2.su2 import ParallelRuleSiteQC, ParallelRuleIdentity
            #OLD_CPX else:
            #OLD_CPX     from block2.sz import ParallelRuleQC, ParallelRuleNPDMQC
            #OLD_CPX     from block2.sz import ParallelRuleSiteQC, ParallelRuleIdentity
            self.prule = bs.ParallelRuleQC(mpi)
            self.pdmrule = bs.ParallelRuleNPDMQC(mpi)
            self.siterule = bs.ParallelRuleSiteQC(mpi)
            self.identrule = bs.ParallelRuleIdentity(mpi)
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

        if spin_symmetry == 'su2':
            n_mo = orbs.shape[1]
            assert orbs.shape[0] == orbs.shape[1]
        elif spin_symmetry == 'sz':
            n_mo = orbs.shape[2]
            assert orbs.shape[1] == orbs.shape[2]
        assert n_mo == self.mol.nao
        n_occ = n_core + n_sites
        n_virt = n_mo - n_occ

        if spin_symmetry == 'su2':
            assert len(orbs.shape) == 2, \
                'If SU2 symmetry is invoked, orbs must be a 2D array.'
            orbs_c = np.zeros((2, n_mo, n_core))
            orbs_s = np.zeros((2, n_mo, n_sites))
            orbs_v = np.zeros((2, n_mo, n_virt))
            for i in range(0,2):
                orbs_c[i,:,:] = orbs[:, 0:n_core]
                orbs_s[i,:,:] = orbs[:, n_core:n_occ]
                orbs_v[i,:,:] = orbs[:, n_occ:n_mo]
        elif spin_symmetry == 'sz':
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
        self.fcidump = bx.FCIDUMP()
        self.fcidump.read(filename)
        self.groupname = pg
        assert self.fcidump.n_elec == sum(self.nel_site), \
            f'init_hamiltonian_fcidump: self.fcidump.n_elec ({self.fcidump.n_elec}) must ' + \
            'be identical to sum(self.nel_site) (%d).' % sum(self.nel_site)

        #==== Reordering indices ====#
        if idx is not None:
            self.fcidump.reorder(b2.VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)

        #==== Orbitals and MPS symemtries ====#
        swap_pg = getattr(b2.PointGroup, "swap_" + pg)
        self.orb_sym = b2.VectorUInt8(map(swap_pg, self.fcidump.orb_sym))      # 1)
        _print("# fcidump symmetrize error:", self.fcidump.symmetrize(orb_sym))
        self.wfn_sym = swap_pg(self.fcidump.isym)
        # NOTE:
        # 1) Because of the self.fcidump.reorder invocation above, self.orb_sym contains
        #    orbital symmetries AFTER REORDERING.

        #==== Construct the Hamiltonian MPO ====#
        vacuum = SX(0)
        self.target = SX(self.fcidump.n_elec, self.fcidump.twos,
                                swap_pg(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites
        self.hamil = bs.HamiltonianQC(
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
        self.fcidump = bx.FCIDUMP()
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
            assert spin_symmetry == 'sz'
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
        self.fcidump.orb_sym = b2.VectorUInt8(orb_sym)       # Hence, self.fcidump.orb_sym is in Molpro convention.

        #==== Reordering indices ====#
        if idx is not None:
            self.fcidump.reorder(b2.VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)

        #==== Orbitals and MPS symemtries ====#
        swap_pg = getattr(b2.PointGroup, "swap_" + pg)
        self.orb_sym = b2.VectorUInt8(map(swap_pg, self.fcidump.orb_sym))      # 1)
        self.wfn_sym = swap_pg(isym)
        # NOTE:
        # 1) Because of the self.fcidump.reorder invocation above, self.orb_sym contains
        #    orbital symmetries AFTER REORDERING.


        #==== Construct the Hamiltonian MPO ====#
        vacuum = SX(0)
        self.target = SX(n_elec, twos, swap_pg(isym))
        self.n_sites = n_sites
        self.hamil = bs.HamiltonianQC(
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
                self.fcidump.orb_sym = b2.VectorUInt8(orb_sym)
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
    def get_one_pdm(self, cpx_mps, mps=None, inmps_name=None, dmargin=0):
        if mps is None and inmps_name is None:
            raise ValueError("The 'mps' and 'inmps_name' parameters of "
                             + "get_one_pdm cannot be both None.")
        
        if self.verbose >= 2:
            _print('>>> START one-pdm <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

        if mps is None:   # mps takes priority over inmps_name, the latter will only be used if the former is None.
            mps_info = brs.MPSInfo(0)
            mps_info.load_data(self.scratch + "/" + inmps_name)
            mps = bw.bs.MPS(mps_info)
            mps.load_data()
            mps.info.load_mutable()
            
        max_bdim = max([x.n_states_total for x in mps.info.left_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim
        max_bdim = max([x.n_states_total for x in mps.info.right_dims])
        if mps.info.bond_dim < max_bdim:
            mps.info.bond_dim = max_bdim

        #OLD_CPX if self.mpi is not None:
        #OLD_CPX     if SpinLabel == SU2:
        #OLD_CPX         from block2.su2 import ParallelMPO
        #OLD_CPX     else:
        #OLD_CPX         from block2.sz import ParallelMPO

        # 1PDM MPO
        pmpo = bs.PDM1MPOQC(self.hamil)
        pmpo = bs.SimplifiedMPO(pmpo, bs.RuleQC())
        if self.mpi is not None:
            pmpo = bs.ParallelMPO(pmpo, self.pdmrule)

        # 1PDM
        pme = bs.MovingEnvironment(pmpo, mps, mps, "1PDM")
        pme.init_environments(False)
        if cpx_mps:
            # WARNING: There is no ComplexExpect in block2.cpx.su2
            expect = b2.su2.ComplexExpect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
        else:
            expect = bs.Expect(pme, mps.info.bond_dim+dmargin, mps.info.bond_dim+dmargin)   #NOTE
        expect.iprint = max(self.verbose - 1, 0)
        expect.solve(True, mps.center == 0)
        if spin_symmetry == 'su2':
            dmr = expect.get_1pdm_spatial(self.n_sites)
            dm = np.array(dmr).copy()
        elif spin_symmetry == 'sz':
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

        if spin_symmetry == 'su2':
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        elif spin_symmetry == 'sz':
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)
    #################################################
#OLD_CPX 
#OLD_CPX 
#OLD_CPX     #################################################
#OLD_CPX     def dmrg(self, bond_dims, noises, n_steps=30, dav_tols=1E-5, conv_tol=1E-7, cutoff=1E-14,
#OLD_CPX              occs=None, bias=1.0, outmps_dir0=None, outmps_name='GS_MPS_INFO',
#OLD_CPX              save_1pdm=False):
#OLD_CPX         """Ground-State DMRG."""
#OLD_CPX 
#OLD_CPX         if self.verbose >= 2:
#OLD_CPX             _print('>>> START GS-DMRG <<<')
#OLD_CPX         t = time.perf_counter()
#OLD_CPX 
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             self.mpi.barrier()
#OLD_CPX         if outmps_dir0 is None:
#OLD_CPX             outmps_dir = self.scratch
#OLD_CPX         else:
#OLD_CPX             outmps_dir = outmps_dir0
#OLD_CPX 
#OLD_CPX         # MultiMPSInfo
#OLD_CPX         mps_info = MPSInfo(self.n_sites, self.hamil.vacuum,
#OLD_CPX                            self.target, self.hamil.basis)
#OLD_CPX         mps_info.tag = 'KET'
#OLD_CPX         if occs is None:
#OLD_CPX             if self.verbose >= 2:
#OLD_CPX                 _print("Using FCI INIT MPS")
#OLD_CPX             mps_info.set_bond_dimension(bond_dims[0])
#OLD_CPX         else:
#OLD_CPX             if self.verbose >= 2:
#OLD_CPX                 _print("Using occupation number INIT MPS")
#OLD_CPX             if self.idx is not None:
#OLD_CPX                 #ERR occs = self.fcidump.reorder(VectorDouble(occs), VectorUInt16(self.idx))
#OLD_CPX                 occs = occs[self.idx]
#OLD_CPX             mps_info.set_bond_dimension_using_occ(
#OLD_CPX                 bond_dims[0], VectorDouble(occs), bias=bias)
#OLD_CPX         
#OLD_CPX         mps = MPS(self.n_sites, 0, 2)   # The 3rd argument controls the use of one/two-site algorithm.
#OLD_CPX         mps.initialize(mps_info)
#OLD_CPX         mps.random_canonicalize()
#OLD_CPX 
#OLD_CPX         mps.save_mutable()
#OLD_CPX         mps.deallocate()
#OLD_CPX         mps_info.save_mutable()
#OLD_CPX         mps_info.deallocate_mutable()
#OLD_CPX         
#OLD_CPX 
#OLD_CPX         # MPO
#OLD_CPX         tx = time.perf_counter()
#OLD_CPX         mpo = MPOQC(self.hamil, QCTypes.Conventional)
#OLD_CPX         mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
#OLD_CPX                             OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX         self.mpo_orig = mpo
#OLD_CPX 
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             if SpinLabel == SU2:
#OLD_CPX                 from block2.su2 import ParallelMPO
#OLD_CPX             else:
#OLD_CPX                 from block2.sz import ParallelMPO
#OLD_CPX             mpo = ParallelMPO(mpo, self.prule)
#OLD_CPX 
#OLD_CPX         if self.verbose >= 3:
#OLD_CPX             _print('MPO time = ', time.perf_counter() - tx)
#OLD_CPX 
#OLD_CPX         if self.print_statistics:
#OLD_CPX             _print('GS MPO BOND DIMS = ', ''.join(
#OLD_CPX                 ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
#OLD_CPX             max_d = max(bond_dims)
#OLD_CPX             mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
#OLD_CPX                                 self.target, self.hamil.basis)
#OLD_CPX             mps_info2.set_bond_dimension(max_d)
#OLD_CPX             _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
#OLD_CPX             _print("GS EST MAX MPS BOND DIMS = ", ''.join(
#OLD_CPX                 ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
#OLD_CPX             _print("GS EST PEAK MEM = ", MYTDDMRG.fmt_size(
#OLD_CPX                 mem2), " SCRATCH = ", MYTDDMRG.fmt_size(disk))
#OLD_CPX             mps_info2.deallocate_mutable()
#OLD_CPX             mps_info2.deallocate()
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         # DMRG
#OLD_CPX         me = MovingEnvironment(mpo, mps, mps, "DMRG")
#OLD_CPX         if self.delayed_contraction:
#OLD_CPX             me.delayed_contraction = OpNamesSet.normal_ops()
#OLD_CPX             me.cached_contraction = True
#OLD_CPX         tx = time.perf_counter()
#OLD_CPX         me.init_environments(self.verbose >= 4)
#OLD_CPX         if self.verbose >= 3:
#OLD_CPX             _print('DMRG INIT time = ', time.perf_counter() - tx)
#OLD_CPX         dmrg = DMRG(me, VectorUBond(bond_dims), VectorDouble(noises))
#OLD_CPX         dmrg.davidson_conv_thrds = VectorDouble(dav_tols)
#OLD_CPX         dmrg.davidson_soft_max_iter = 4000
#OLD_CPX         dmrg.noise_type = NoiseTypes.ReducedPerturbative
#OLD_CPX         dmrg.decomp_type = DecompositionTypes.SVD
#OLD_CPX         dmrg.iprint = max(self.verbose - 1, 0)
#OLD_CPX         dmrg.cutoff = cutoff
#OLD_CPX         dmrg.solve(n_steps, mps.center == 0, conv_tol)
#OLD_CPX 
#OLD_CPX         self.gs_energy = dmrg.energies[-1][0]
#OLD_CPX         self.bond_dim = bond_dims[-1]
#OLD_CPX         _print("Ground state energy = %20.15f" % self.gs_energy)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== MO occupations ====#
#OLD_CPX         dm0 = self.get_one_pdm(False, mps)
#OLD_CPX         dm0_full = make_full_dm(self.n_core, dm0)
#OLD_CPX         occs0 = np.zeros((2, self.n_core+self.n_sites))
#OLD_CPX         for i in range(0, 2): occs0[i,:] = np.diag(dm0_full[i,:,:]).copy()
#OLD_CPX         print_orb_occupations(occs0)
#OLD_CPX         
#OLD_CPX             
#OLD_CPX         #==== Partial charge ====#
#OLD_CPX         orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()), axis=2)
#OLD_CPX         self.qmul0, self.qlow0 = \
#OLD_CPX             pcharge.calc(self.mol, dm0_full, orbs, self.ovl_ao)
#OLD_CPX         print_pcharge(self.mol, self.qmul0, self.qlow0)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Multipole analysis ====#
#OLD_CPX         e_dpole, n_dpole, e_qpole, n_qpole = \
#OLD_CPX             mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm0_full, orbs)
#OLD_CPX         print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)
#OLD_CPX 
#OLD_CPX         
#OLD_CPX         #==== Save the output MPS ====#
#OLD_CPX         #OLD mps.save_data()
#OLD_CPX         #OLD mps_info.save_data(self.scratch + "/GS_MPS_INFO")
#OLD_CPX         #OLD mps_info.deallocate()
#OLD_CPX         _print('')
#OLD_CPX         _print('Saving the ground state MPS files under ' + outmps_dir)
#OLD_CPX         if outmps_dir != self.scratch:
#OLD_CPX             mkDir(outmps_dir)
#OLD_CPX         mps_info.save_data(outmps_dir + "/" + outmps_name)
#OLD_CPX         saveMPStoDir(mps, outmps_dir, self.mpi)
#OLD_CPX         _print('Output ground state max. bond dimension = ', mps.info.bond_dim)
#OLD_CPX         if save_1pdm:
#OLD_CPX             np.save(outmps_dir + '/GS_1pdm', dm0)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Statistics ====#
#OLD_CPX         if self.print_statistics:
#OLD_CPX             dmain, dseco, imain, iseco = Global.frame.peak_used_memory
#OLD_CPX             _print("GS PEAK MEM USAGE:",
#OLD_CPX                    "DMEM = ", MYTDDMRG.fmt_size(dmain + dseco),
#OLD_CPX                    "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
#OLD_CPX                    "IMEM = ", MYTDDMRG.fmt_size(imain + iseco),
#OLD_CPX                    "(%.0f%%)" % (imain * 100 / (imain + iseco)))
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         if self.verbose >= 2:
#OLD_CPX             _print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
#OLD_CPX                    (time.perf_counter() - t))
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX     
#OLD_CPX     #################################################
#OLD_CPX     def save_gs_mps(self, save_dir='./gs_mps'):
#OLD_CPX         import shutil
#OLD_CPX         import pickle
#OLD_CPX         import os
#OLD_CPX         if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX             pickle.dump(self.gs_energy, open(
#OLD_CPX                 self.scratch + '/GS_ENERGY', 'wb'))
#OLD_CPX             for k in os.listdir(self.scratch):
#OLD_CPX                 if '.KET.' in k or k == 'GS_MPS_INFO' or k == 'GS_ENERGY':
#OLD_CPX                     shutil.copy(self.scratch + "/" + k, save_dir + "/" + k)
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             self.mpi.barrier()
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX     
#OLD_CPX     #################################################
#OLD_CPX     def load_gs_mps(self, load_dir='./gs_mps'):
#OLD_CPX         import shutil
#OLD_CPX         import pickle
#OLD_CPX         import os
#OLD_CPX         if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX             for k in os.listdir(load_dir):
#OLD_CPX                 shutil.copy(load_dir + "/" + k, self.scratch + "/" + k)
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             self.mpi.barrier()
#OLD_CPX         self.gs_energy = pickle.load(open(self.scratch + '/GS_ENERGY', 'rb'))
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX 
#OLD_CPX     #################################################
#OLD_CPX     def print_occupation_table(self, dm, aorb):
#OLD_CPX 
#OLD_CPX         from pyscf import symm
#OLD_CPX         
#OLD_CPX         natocc_a = eigvalsh(dm[0,:,:])
#OLD_CPX         natocc_a = natocc_a[::-1]
#OLD_CPX         natocc_b = eigvalsh(dm[1,:,:])
#OLD_CPX         natocc_b = natocc_b[::-1]
#OLD_CPX 
#OLD_CPX         mk = ''
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             mk = ' (ann.)'
#OLD_CPX         ll = 4 + 16 + 15 + 12 + (0 if isinstance(aorb, int) else 13) + \
#OLD_CPX              (3+18) + 18 + 2*len(mk)
#OLD_CPX         hline = ''.join(['-' for i in range(0, ll)])
#OLD_CPX         aspace = ''.join([' ' for i in range(0,len(mk))])
#OLD_CPX 
#OLD_CPX         _print(hline)
#OLD_CPX         _print('%4s'  % 'No.', end='') 
#OLD_CPX         _print('%16s' % 'Alpha MO occ.' + aspace, end='')
#OLD_CPX         _print('%15s' % 'Beta MO occ.' + aspace,  end='')
#OLD_CPX         _print('%13s' % 'Irrep / ID',  end='')
#OLD_CPX         if isinstance(aorb, np.ndarray):
#OLD_CPX             _print('%13s' % 'aorb coeff', end='')
#OLD_CPX         _print('   %18s' % 'Alpha natorb occ.', end='')
#OLD_CPX         _print('%18s' % 'Beta natorb occ.',  end='\n')
#OLD_CPX         _print(hline)
#OLD_CPX 
#OLD_CPX         for i in range(0, dm.shape[1]):
#OLD_CPX             if isinstance(aorb, int):
#OLD_CPX                 mk0 = aspace
#OLD_CPX                 if i == aorb: mk0 = mk
#OLD_CPX             else:
#OLD_CPX                 mk0 = aspace
#OLD_CPX 
#OLD_CPX             _print('%4d' % i, end='')
#OLD_CPX             _print('%16.8f' % np.diag(dm[0,:,:])[i] + mk0, end='')
#OLD_CPX             _print('%15.8f' % np.diag(dm[1,:,:])[i] + mk0, end='')
#OLD_CPX             j = i if self.ridx is None else self.ridx[i]
#OLD_CPX             sym_label = symm.irrep_id2name(self.groupname, self.orb_sym[j])
#OLD_CPX             _print('%13s' % (sym_label + ' / ' + str(self.orb_sym[j])), end='')
#OLD_CPX             if isinstance(aorb, np.ndarray):
#OLD_CPX                 _print('%13.8f' % aorb[i], end='')
#OLD_CPX             _print('   %18.8f' % natocc_a[i], end='')
#OLD_CPX             _print('%18.8f' % natocc_b[i], end='\n')
#OLD_CPX 
#OLD_CPX         _print(hline)
#OLD_CPX         _print('%4s' % 'Sum', end='')
#OLD_CPX         _print('%16.8f' % np.trace(dm[0,:,:]) + aspace, end='')
#OLD_CPX         _print('%15.8f' % np.trace(dm[1,:,:]) + aspace, end='')
#OLD_CPX         _print('%13s' % ' ', end='')
#OLD_CPX         if isinstance(aorb, np.ndarray):
#OLD_CPX             _print('%13s' % ' ', end='')
#OLD_CPX         _print('   %18.8f' % sum(natocc_a), end='')
#OLD_CPX         _print('%18.8f' % sum(natocc_b), end='\n')
#OLD_CPX         _print(hline)
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX     
#OLD_CPX     #################################################
#OLD_CPX     def annihilate(self, aorb, fit_bond_dims, fit_noises, fit_conv_tol, fit_n_steps, pg,
#OLD_CPX                    inmps_dir0=None, inmps_name='GS_MPS_INFO', outmps_dir0=None,
#OLD_CPX                    outmps_name='ANN_KET', aorb_thr=1.0E-12, alpha=True, 
#OLD_CPX                    cutoff=1E-14, occs=None, bias=1.0, outmps_normal=True, save_1pdm=False):
#OLD_CPX         """
#OLD_CPX         aorb can be int, numpy.ndarray, or 'nat<n>' where n is an integer'
#OLD_CPX         """
#OLD_CPX         ##OLD ops = [None] * len(aorb)
#OLD_CPX         ##OLD rkets = [None] * len(aorb)
#OLD_CPX         ##OLD rmpos = [None] * len(aorb)
#OLD_CPX         from pyscf import symm
#OLD_CPX 
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             self.mpi.barrier()
#OLD_CPX 
#OLD_CPX         if inmps_dir0 is None:
#OLD_CPX             inmps_dir = self.scratch
#OLD_CPX         else:
#OLD_CPX             inmps_dir = inmps_dir0
#OLD_CPX         if outmps_dir0 is None:
#OLD_CPX             outmps_dir = self.scratch
#OLD_CPX         else:
#OLD_CPX             outmps_dir = outmps_dir0
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         #==== Checking input parameters ====#
#OLD_CPX         if not (isinstance(aorb, int) or isinstance(aorb, np.ndarray) or
#OLD_CPX                 isinstance(aorb, str)):
#OLD_CPX             raise ValueError('The argument \'aorb\' of MYTDDMRG.annihilate method must ' +
#OLD_CPX                              'be either an integer, a numpy.ndarray, or a string. ' +
#OLD_CPX                              f'Currently, aorb = {aorb}.')
#OLD_CPX 
#OLD_CPX         use_natorb = False
#OLD_CPX         if isinstance(aorb, str):
#OLD_CPX             nn = len(aorb)
#OLD_CPX             ss = aorb[0:3]
#OLD_CPX             ss_ = aorb[3:nn]
#OLD_CPX             if ss != 'nat' or not ss_.isdigit:
#OLD_CPX                 _print('aorb = ', aorb)
#OLD_CPX                 raise ValueError(
#OLD_CPX                     'The only way the argument \'aorb\' of MYTDDMRG.annihilate ' +
#OLD_CPX                     'can be a string is when it has the format of \'nat<n>\', ' +
#OLD_CPX                     'where \'n\' must be an integer, e.g. \'nat1\', \'nat24\', ' +
#OLD_CPX                     f'etc. Currently aorb = {aorb:s}')
#OLD_CPX             nat_id = int(ss_)
#OLD_CPX             if nat_id < 0 or nat_id >= self.n_sites:
#OLD_CPX                 raise ValueError('The index of the natural orbital specified by ' +
#OLD_CPX                                  'the argument \'aorb\' of MYTDDMRG.annihilate ' +
#OLD_CPX                                  'is out of bound, which is between 0 and ' +
#OLD_CPX                                  f'{self.n_sites:d}. Currently, the specified index ' +
#OLD_CPX                                  f'is {nat_id:d}.')
#OLD_CPX             use_natorb = True
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #cpx bw.bs.SimplifiedMPO
#OLD_CPX         #cpx bw.bs.RuleQC
#OLD_CPX         #cpx bw.bs.IdentityMPO
#OLD_CPX         #cpx bw.bs.ParallelMPO
#OLD_CPX         #cpx consider DMRGDriver.get_identity_mpo
#OLD_CPX         idMPO_ = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             idMPO_ = ParallelMPO(idMPO_, self.identrule)
#OLD_CPX 
#OLD_CPX         
#OLD_CPX 
#OLD_CPX         #cpx bw.bs.ParallelMPO
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             if SpinLabel == SU2:
#OLD_CPX                 from block2.su2 import ParallelMPO
#OLD_CPX             else:
#OLD_CPX                 from block2.sz import ParallelMPO
#OLD_CPX 
#OLD_CPX                 
#OLD_CPX         #==== Build Hamiltonian MPO (to provide noise ====#
#OLD_CPX         #====      for Linear.solve down below)       ====#
#OLD_CPX         if self.mpo_orig is None:
#OLD_CPX             mpo = MPOQC(self.hamil, QCTypes.Conventional)
#OLD_CPX             mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
#OLD_CPX                                 OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX             self.mpo_orig = mpo
#OLD_CPX 
#OLD_CPX         mpo = 1.0 * self.mpo_orig
#OLD_CPX         print_MPO_bond_dims(mpo, 'Hamiltonian')
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             mpo = ParallelMPO(mpo, self.prule)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Load the input MPS ====#
#OLD_CPX         inmps_path = inmps_dir + "/" + inmps_name
#OLD_CPX         _print('Loading input MPS info from ' + inmps_path)
#OLD_CPX         mps_info = MPSInfo(0)
#OLD_CPX         mps_info.load_data(inmps_path)
#OLD_CPX         #OLD mps_info.load_data(self.scratch + "/GS_MPS_INFO")
#OLD_CPX         #OLD mps = MPS(mps_info)
#OLD_CPX         #OLD mps.load_data()
#OLD_CPX         mps = loadMPSfromDir(mps_info, inmps_dir, self.mpi)
#OLD_CPX         _print('Input MPS max. bond dimension = ', mps.info.bond_dim)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Compute the requested natural orbital ====#
#OLD_CPX         dm0 = self.get_one_pdm(False, mps)
#OLD_CPX         if use_natorb:
#OLD_CPX             e0, aorb = eigh(dm0[0 if alpha else 1,:,:])
#OLD_CPX             aorb = aorb[:,::-1]       # Reverse the columns
#OLD_CPX             aorb = aorb[:,nat_id]
#OLD_CPX             aorb[np.abs(aorb) < aorb_thr] = 0.0          # 1)
#OLD_CPX             _print(aorb)
#OLD_CPX         _print('Occupations before annihilation:')
#OLD_CPX         self.print_occupation_table(dm0, aorb)
#OLD_CPX         # NOTES:
#OLD_CPX         # 1) For some reason, without setting the small coefficients to zero,
#OLD_CPX         #    a segfault error happens later inside the MPS_fitting function.
#OLD_CPX 
#OLD_CPX         
#OLD_CPX         #==== Some statistics ====#
#OLD_CPX         if self.print_statistics:
#OLD_CPX             max_d = max(fit_bond_dims)
#OLD_CPX             mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
#OLD_CPX                                 self.target, self.hamil.basis)
#OLD_CPX             mps_info2.set_bond_dimension(max_d)
#OLD_CPX             _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
#OLD_CPX             _print("EST MAX OUTPUT MPS BOND DIMS = ", ''.join(
#OLD_CPX                 ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
#OLD_CPX             _print("EST PEAK MEM = ", MYTDDMRG.fmt_size(mem2),
#OLD_CPX                    " SCRATCH = ", MYTDDMRG.fmt_size(disk))
#OLD_CPX             mps_info2.deallocate_mutable()
#OLD_CPX             mps_info2.deallocate()
#OLD_CPX 
#OLD_CPX         #NOTE: check if ridx is not none
#OLD_CPX 
#OLD_CPX         
#OLD_CPX         #==== Determine the reordered index of the annihilated orbital ====#
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             idx = aorb if self.ridx is None else self.ridx[aorb]
#OLD_CPX             # idx is the index of the annihilated orbital after reordering.
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             if self.idx is not None:
#OLD_CPX                 aorb = aorb[self.idx]            
#OLD_CPX             
#OLD_CPX             #== Determine the irrep. of aorb ==#
#OLD_CPX             for i in range(0, self.n_sites):
#OLD_CPX                 j = i if self.ridx is None else self.ridx[i]
#OLD_CPX                 if (np.abs(aorb[j]) >= aorb_thr):
#OLD_CPX                     aorb_sym = self.orb_sym[j]            # 1)
#OLD_CPX                     break
#OLD_CPX             for i in range(0, self.n_sites):
#OLD_CPX                 j = i if self.ridx is None else self.ridx[i]
#OLD_CPX                 if (np.abs(aorb[j]) >= aorb_thr and self.orb_sym[j] != aorb_sym):
#OLD_CPX                     _print(self.orb_sym)
#OLD_CPX                     _print('An inconsistency in the orbital symmetry found in aorb: ')
#OLD_CPX                     _print(f'   The first detected nonzero element has a symmetry ID of {aorb_sym:d}, ', end='')
#OLD_CPX                     _print(f'but the symmetry ID of another nonzero element (the {i:d}-th element) ', end='')
#OLD_CPX                     _print(f'is {self.orb_sym[j]:d}.')
#OLD_CPX                     raise ValueError('The orbitals making the linear combination in ' +
#OLD_CPX                                      'aorb must all have the same symmetry.')
#OLD_CPX             # NOTES:
#OLD_CPX             # 1) j instead of i is used as the index for self.orb_sym because the
#OLD_CPX             #    contents of this array have been reordered (see its assignment in the
#OLD_CPX             #    self.init_hamiltonian or self.init_hamiltonian_fcidump function).
#OLD_CPX 
#OLD_CPX                 
#OLD_CPX         #==== Begin constructing the annihilation MPO ====#
#OLD_CPX         gidxs = list(range(self.n_sites))
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             if SpinLabel == SZ:
#OLD_CPX                 ops = OpElement(OpNames.D, SiteIndex((idx, ), (0 if alpha else 1, )), 
#OLD_CPX                     SZ(-1, -1 if alpha else 1, self.orb_sym[idx]))
#OLD_CPX             else:
#OLD_CPX                 ops = OpElement(OpNames.D, SiteIndex((idx, ), ()), 
#OLD_CPX                     SU2(-1, 1, self.orb_sym[idx]))
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             ops = [None] * self.n_sites
#OLD_CPX             for ii, ix in enumerate(gidxs):
#OLD_CPX                 if SpinLabel == SZ:
#OLD_CPX                     ops[ii] = OpElement(OpNames.D, SiteIndex((ix, ), (0 if alpha else 1, )),
#OLD_CPX                         SZ(-1, -1 if alpha else 1, aorb_sym))
#OLD_CPX                 else:
#OLD_CPX                     ops[ii] = OpElement(OpNames.D, SiteIndex((ix, ), ()), 
#OLD_CPX                         SU2(-1, 1, aorb_sym))
#OLD_CPX                     
#OLD_CPX 
#OLD_CPX         #==== Determine if the annihilated orbital is a site orbital ====#
#OLD_CPX         #====  or a linear combination of them (orbital transform)   ====#
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             rmpos = SimplifiedMPO(
#OLD_CPX                 SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             ao_ops = VectorOpElement([None] * self.n_sites)
#OLD_CPX             for ix in range(self.n_sites):
#OLD_CPX                 ao_ops[ix] = ops[ix] * aorb[ix]
#OLD_CPX                 _print('opsx = ', ops[ix], type(ops[ix]), ao_ops[ix], type(ao_ops[ix]))
#OLD_CPX             rmpos = SimplifiedMPO(
#OLD_CPX                 LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             rmpos = ParallelMPO(rmpos, self.siterule)
#OLD_CPX 
#OLD_CPX                                 
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             self.mpi.barrier()
#OLD_CPX         if self.verbose >= 2:
#OLD_CPX             _print('>>> START : Applying the annihilation operator <<<')
#OLD_CPX         t = time.perf_counter()
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Determine the quantum numbers of the output MPS, rkets ====#
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             ion_target = self.target + ops.q_label
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             ion_sym = self.wfn_sym ^ aorb_sym
#OLD_CPX             ion_target = SpinLabel(sum(self.nel_site)-1, 1, ion_sym)
#OLD_CPX         rket_info = MPSInfo(self.n_sites, self.hamil.vacuum, ion_target, self.hamil.basis)
#OLD_CPX         
#OLD_CPX         _print('Quantum number information:')
#OLD_CPX         _print(' - Input MPS = ', self.target)
#OLD_CPX         _print(' - Input MPS multiplicity = ', self.target.multiplicity)
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             _print(' - Annihilated orbital = ', ops.q_label)
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             _print(' - Annihilated orbital = ', SpinLabel(-1, 1, aorb_sym))
#OLD_CPX         _print(' - Output MPS = ', ion_target)
#OLD_CPX         _print(' - Output MPS multiplicity = ', ion_target.multiplicity)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Tag the output MPS ====#
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             rket_info.tag = 'DKET_%d' % idx
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             rket_info.tag = 'DKET_C'
#OLD_CPX         
#OLD_CPX 
#OLD_CPX         #==== Set the bond dimension of output MPS ====#
#OLD_CPX         rket_info.set_bond_dimension(mps.info.bond_dim)
#OLD_CPX         if occs is None:
#OLD_CPX             if self.verbose >= 2:
#OLD_CPX                 _print("Using FCI INIT MPS")
#OLD_CPX             rket_info.set_bond_dimension(mps.info.bond_dim)
#OLD_CPX         else:
#OLD_CPX             if self.verbose >= 2:
#OLD_CPX                 _print("Using occupation number INIT MPS")
#OLD_CPX             if self.idx is not None:
#OLD_CPX                 occs = occs[self.idx]
#OLD_CPX             rket_info.set_bond_dimension_using_occ(
#OLD_CPX                 mps.info.bond_dim, VectorDouble(occs), bias=bias)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Initialization of output MPS ====#
#OLD_CPX         rket_info.save_data(self.scratch + "/" + outmps_name)
#OLD_CPX         rkets = MPS(self.n_sites, mps.center, 2)
#OLD_CPX         rkets.initialize(rket_info)
#OLD_CPX         rkets.random_canonicalize()
#OLD_CPX         rkets.save_mutable()
#OLD_CPX         rkets.deallocate()
#OLD_CPX         rket_info.save_mutable()
#OLD_CPX         rket_info.deallocate_mutable()
#OLD_CPX 
#OLD_CPX         #OLD if mo_coeff is None:
#OLD_CPX         #OLD     # the mpo and gf are in the same basis
#OLD_CPX         #OLD     # the mpo is SiteMPO
#OLD_CPX         #OLD     rmpos = SimplifiedMPO(
#OLD_CPX         #OLD         SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX         #OLD else:
#OLD_CPX         #OLD     # the mpo is in mo basis and gf is in ao basis
#OLD_CPX         #OLD     # the mpo is sum of SiteMPO (LocalMPO)
#OLD_CPX         #OLD     ao_ops = VectorOpElement([None] * self.n_sites)
#OLD_CPX         #OLD     _print('mo_coeff = ', mo_coeff)
#OLD_CPX         #OLD     for ix in range(self.n_sites):
#OLD_CPX         #OLD         ao_ops[ix] = ops[ix] * mo_coeff[ix]
#OLD_CPX         #OLD         _print('opsx = ', ops[ix], type(ops[ix]), ao_ops[ix], type(ao_ops[ix]))
#OLD_CPX         #OLD     rmpos = SimplifiedMPO(
#OLD_CPX         #OLD         LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX         #OLD     
#OLD_CPX         #OLD if self.mpi is not None:
#OLD_CPX         #OLD     rmpos = ParallelMPO(rmpos, self.siterule)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Solve for the output MPS ====#
#OLD_CPX         MPS_fitting(rkets, mps, rmpos, fit_bond_dims, fit_n_steps, fit_noises,
#OLD_CPX                     fit_conv_tol, 'density_mat', cutoff, lmpo=mpo,
#OLD_CPX                     verbose_lvl=self.verbose-1)
#OLD_CPX         _print('Output MPS max. bond dimension = ', rkets.info.bond_dim)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Normalize the output MPS if requested ====#
#OLD_CPX         if outmps_normal:
#OLD_CPX             _print('Normalizing the output MPS')
#OLD_CPX             icent = rkets.center
#OLD_CPX             #OLD if rkets.dot == 2 and rkets.center == rkets.n_sites-1:
#OLD_CPX             if rkets.dot == 2:
#OLD_CPX                 if rkets.center == rkets.n_sites-2:
#OLD_CPX                     icent += 1
#OLD_CPX                 elif rkets.center == 0:
#OLD_CPX                     pass
#OLD_CPX             assert rkets.tensors[icent] is not None
#OLD_CPX             
#OLD_CPX             rkets.load_tensor(icent)
#OLD_CPX             rkets.tensors[icent].normalize()
#OLD_CPX             rkets.save_tensor(icent)
#OLD_CPX             rkets.unload_tensor(icent)
#OLD_CPX             # rket_info.save_data(self.scratch + "/" + outmps_name)
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         #==== Check the norm ====#
#OLD_CPX         idMPO_ = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             idMPO_ = ParallelMPO(idMPO_, self.identrule)
#OLD_CPX         idN = MovingEnvironment(idMPO_, rkets, rkets, "norm")
#OLD_CPX         idN.init_environments()
#OLD_CPX         nrm = Expect(idN, rkets.info.bond_dim, rkets.info.bond_dim)
#OLD_CPX         nrm_ = nrm.solve(False)
#OLD_CPX         _print('Output MPS norm = %11.8f' % nrm_)
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         #==== Print the energy of the output MPS ====#
#OLD_CPX         energy = calc_energy_MPS(mpo, rkets, 0)
#OLD_CPX         _print('Output MPS energy = %12.8f Hartree' % energy)
#OLD_CPX         _print('Canonical form of the annihilation output = ', rkets.canonical_form)
#OLD_CPX         dm1 = self.get_one_pdm(False, rkets)
#OLD_CPX         _print('Occupations after annihilation:')
#OLD_CPX         if isinstance(aorb, int):
#OLD_CPX             self.print_occupation_table(dm1, aorb)
#OLD_CPX         elif isinstance(aorb, np.ndarray):
#OLD_CPX             if self.ridx is None:
#OLD_CPX                 self.print_occupation_table(dm1, aorb)
#OLD_CPX             else:
#OLD_CPX                 self.print_occupation_table(dm1, aorb[self.ridx])
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         #==== Partial charge ====#
#OLD_CPX         dm1_full = make_full_dm(self.n_core, dm1)
#OLD_CPX         orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()), axis=2)
#OLD_CPX         self.qmul1, self.qlow1 = \
#OLD_CPX             pcharge.calc(self.mol, dm1_full, orbs, self.ovl_ao)
#OLD_CPX         print_pcharge(self.mol, self.qmul1, self.qlow1)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Multipole analysis ====#
#OLD_CPX         e_dpole, n_dpole, e_qpole, n_qpole = \
#OLD_CPX             mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm1_full, orbs)
#OLD_CPX         print_mpole(e_dpole, n_dpole, e_qpole, n_qpole)
#OLD_CPX 
#OLD_CPX         
#OLD_CPX         #==== Save the output MPS ====#
#OLD_CPX         _print('')
#OLD_CPX         if outmps_dir != self.scratch:
#OLD_CPX             mkDir(outmps_dir)
#OLD_CPX         rket_info.save_data(outmps_dir + "/" + outmps_name)
#OLD_CPX         _print('Saving output MPS files under ' + outmps_dir)
#OLD_CPX         saveMPStoDir(rkets, outmps_dir, self.mpi)
#OLD_CPX         if save_1pdm:
#OLD_CPX             _print('Saving 1PDM of the output MPS under ' + outmps_dir)
#OLD_CPX             np.save(outmps_dir + '/ANN_1pdm', dm1)
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         if self.verbose >= 2:
#OLD_CPX             _print('>>> COMPLETE : Application of annihilation operator | Time = %.2f <<<' %
#OLD_CPX                    (time.perf_counter() - t))
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX 
#OLD_CPX     #################################################
#OLD_CPX     def save_time_info(self, save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps,
#OLD_CPX                        save_1pdm, dm):
#OLD_CPX 
#OLD_CPX         def save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps,
#OLD_CPX                             save_1pdm, dm):
#OLD_CPX             yn_bools = ('No','Yes')
#OLD_CPX             au2fs = 2.4188843265e-2   # a.u. of time to fs conversion factor
#OLD_CPX             with open(save_dir + '/TIME_INFO', 'w') as t_info:
#OLD_CPX                 t_info.write(' Actual sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
#OLD_CPX                              (it, t, t*au2fs))
#OLD_CPX                 t_info.write(' Requested sampling time = (%d, %10.6f a.u. / %10.6f fs)\n' %
#OLD_CPX                              (i_sp, t_sp, t_sp*au2fs))
#OLD_CPX                 t_info.write(' MPS norm square = %19.14f\n' % normsq)
#OLD_CPX                 t_info.write(' Autocorrelation = (%19.14f, %19.14f)\n' % (ac.real, ac.imag))
#OLD_CPX                 t_info.write(f' Is MPS saved at this time?  {yn_bools[save_mps]}\n')
#OLD_CPX                 t_info.write(f' Is 1PDM saved at this time?  {yn_bools[save_1pdm]}\n')
#OLD_CPX                 if save_1pdm:
#OLD_CPX                     natocc_a = eigvalsh(dm[0,:,:])
#OLD_CPX                     natocc_b = eigvalsh(dm[1,:,:])
#OLD_CPX                     t_info.write(' 1PDM info:\n')
#OLD_CPX                     t_info.write('    Trace (alpha,beta) = (%16.12f,%16.12f) \n' %
#OLD_CPX                                  ( np.trace(dm[0,:,:]).real, np.trace(dm[1,:,:]).real ))
#OLD_CPX                     t_info.write('    ')
#OLD_CPX                     for i in range(0, 4+4*20): t_info.write('-')
#OLD_CPX                     t_info.write('\n')
#OLD_CPX                     t_info.write('    ' +
#OLD_CPX                                  '%4s'  % 'No.' + 
#OLD_CPX                                  '%20s' % 'Alpha MO occ.' +
#OLD_CPX                                  '%20s' % 'Beta MO occ.' +
#OLD_CPX                                  '%20s' % 'Alpha natorb occ.' +
#OLD_CPX                                  '%20s' % 'Beta natorb occ.' + '\n')
#OLD_CPX                     t_info.write('    ')
#OLD_CPX                     for i in range(0, 4+4*20): t_info.write('-')
#OLD_CPX                     t_info.write('\n')
#OLD_CPX                     for i in range(0, dm.shape[1]):
#OLD_CPX                         t_info.write('    ' +
#OLD_CPX                                      '%4d'  % i + 
#OLD_CPX                                      '%20.12f' % np.diag(dm[0,:,:])[i].real +
#OLD_CPX                                      '%20.12f' % np.diag(dm[1,:,:])[i].real +
#OLD_CPX                                      '%20.12f' % natocc_a[i].real +
#OLD_CPX                                      '%20.12f' % natocc_b[i].real + '\n')
#OLD_CPX                 
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             if self.mpi.rank == 0:
#OLD_CPX                 save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps, save_1pdm,
#OLD_CPX                                 dm)
#OLD_CPX             self.mpi.barrier()
#OLD_CPX         else:
#OLD_CPX             save_time_info0(save_dir, t, it, t_sp, i_sp, normsq, ac, save_mps, save_1pdm, dm)
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX 
#OLD_CPX     #################################################
#OLD_CPX     def get_te_times(self, dt0, tmax):
#OLD_CPX 
#OLD_CPX         #OLD n_steps = int(tmax/dt + 1)
#OLD_CPX         #OLD ts = np.linspace(0, tmax, n_steps)    # times
#OLD_CPX         if type(dt0) is not list:
#OLD_CPX             dt = [dt0]
#OLD_CPX         else:
#OLD_CPX             dt = dt0
#OLD_CPX         ts = [0.0]
#OLD_CPX         i = 1
#OLD_CPX         while ts[-1] < tmax:
#OLD_CPX             if i <= len(dt):
#OLD_CPX                 ts = ts + [sum(dt[0:i])]
#OLD_CPX             else:
#OLD_CPX                 ts = ts + [ts[-1] + dt[-1]]
#OLD_CPX             i += 1
#OLD_CPX         if ts[-1] > tmax:
#OLD_CPX             ts[-1] = tmax
#OLD_CPX             if abs(ts[-1]-ts[-2]) < 1E-3:
#OLD_CPX                 ts.pop()
#OLD_CPX                 ts[-1] = tmax
#OLD_CPX         ts = np.array(ts)
#OLD_CPX         return ts
#OLD_CPX     #################################################
#OLD_CPX 
#OLD_CPX 
#OLD_CPX     ##################################################################
#OLD_CPX     def time_propagate(self, max_bond_dim: int, method, tmax: float, dt0: float, 
#OLD_CPX                        inmps_dir0=None, inmps_name='ANN_KET', exp_tol=1e-6, cutoff=0, 
#OLD_CPX                        normalize=False, n_sub_sweeps=2, n_sub_sweeps_init=4, krylov_size=20, 
#OLD_CPX                        krylov_tol=5.0E-6, t_sample=None, save_mps=False, save_1pdm=False, 
#OLD_CPX                        save_2pdm=False, sample_dir='samples', prefix='te', save_txt=True,
#OLD_CPX                        save_npy=False, prefit=False, prefit_bond_dims=None, 
#OLD_CPX                        prefit_nsteps=None, prefit_noises=None, prefit_conv_tol=None, 
#OLD_CPX                        prefit_cutoff=None, verbosity=6):
#OLD_CPX         '''
#OLD_CPX         Coming soon
#OLD_CPX         '''
#OLD_CPX 
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             if SpinLabel == SU2:
#OLD_CPX                 from block2.su2 import ParallelMPO
#OLD_CPX             else:
#OLD_CPX                 from block2.sz import ParallelMPO
#OLD_CPX 
#OLD_CPX         if inmps_dir0 is None:
#OLD_CPX             inmps_dir = self.scratch
#OLD_CPX         else:
#OLD_CPX             inmps_dir = inmps_dir0
#OLD_CPX 
#OLD_CPX             
#OLD_CPX         #==== Construct the time vector ====#
#OLD_CPX         ts = self.get_te_times(dt0, tmax)
#OLD_CPX         n_steps = len(ts)
#OLD_CPX         _print('Time points (a.u.) = ', ts)
#OLD_CPX             
#OLD_CPX         #==== Initiate the computations and printings of observables ====#
#OLD_CPX         ac_print = print_autocorrelation(prefix, len(ts), save_txt, save_npy)
#OLD_CPX         if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX             ac_print.header()
#OLD_CPX         
#OLD_CPX         #==== Initiate Lowdin partial charges file ====#
#OLD_CPX         if t_sample is not None:
#OLD_CPX             atom_symbol = [self.mol.atom_symbol(i) for i in range(0, self.mol.natm)]
#OLD_CPX             q_print = print_td_pcharge(atom_symbol, prefix, len(t_sample), 8, 
#OLD_CPX                                             save_txt, save_npy)
#OLD_CPX             if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX                 q_print.header()
#OLD_CPX             if self.mpi is not None: self.mpi.barrier()
#OLD_CPX 
#OLD_CPX         #==== Initiate multipole components file ====#
#OLD_CPX         if t_sample is not None:
#OLD_CPX             mp_print = print_td_mpole(prefix, len(t_sample), save_txt, save_npy)
#OLD_CPX             if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX                 mp_print.header()
#OLD_CPX             if self.mpi is not None: self.mpi.barrier()
#OLD_CPX                 
#OLD_CPX         #==== Prepare Hamiltonian MPO ====#
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             self.mpi.barrier()
#OLD_CPX         if self.mpo_orig is None:
#OLD_CPX             mpo = MPOQC(self.hamil, QCTypes.Conventional)
#OLD_CPX             mpo = SimplifiedMPO(mpo, RuleQC(), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
#OLD_CPX             self.mpo_orig = mpo
#OLD_CPX         mpo = 1.0 * self.mpo_orig
#OLD_CPX         print_MPO_bond_dims(mpo, 'Hamiltonian')
#OLD_CPX         #need? mpo = IdentityAddedMPO(mpo) # hrl: alternative
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             mpo = ParallelMPO(mpo, self.prule)
#OLD_CPX 
#OLD_CPX         #==== Load the initial MPS ====#
#OLD_CPX         inmps_path = inmps_dir + "/" + inmps_name
#OLD_CPX         _print('Loading initial MPS info from ' + inmps_path)
#OLD_CPX         mps_info = MPSInfo(0)
#OLD_CPX         mps_info.load_data(inmps_path)
#OLD_CPX         nel_t0 = mps_info.target.n + self.nel_core
#OLD_CPX         #OLD mps = MPS(mps_info)       # This MPS-loading way does not allow loading from directories other than scratch.
#OLD_CPX         #OLD mps.load_data()           # This MPS-loading way does not allow loading from directories other than scratch.
#OLD_CPX         #OLD mps.info.load_mutable()   # This MPS-loading way does not allow loading from directories other than scratch.
#OLD_CPX         mps = loadMPSfromDir(mps_info, inmps_dir, self.mpi)
#OLD_CPX 
#OLD_CPX         #==== Initial norm ====#
#OLD_CPX         idMPO = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
#OLD_CPX         print_MPO_bond_dims(idMPO, 'Identity_2')
#OLD_CPX         if self.mpi is not None:
#OLD_CPX             idMPO = ParallelMPO(idMPO, self.identrule)
#OLD_CPX         idN = MovingEnvironment(idMPO, mps, mps, "norm_in")
#OLD_CPX         idN.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
#OLD_CPX         nrm = Expect(idN, mps.info.bond_dim, mps.info.bond_dim)
#OLD_CPX         nrm_ = nrm.solve(False)
#OLD_CPX         _print(f'Initial MPS norm = {nrm_:11.8f}')
#OLD_CPX         
#OLD_CPX         #==== If a change of bond dimension of the initial MPS is requested ====#
#OLD_CPX         if prefit:
#OLD_CPX             if self.mpi is not None: self.mpi.barrier()
#OLD_CPX             ref_mps = mps.deep_copy('ref_mps_t0')
#OLD_CPX             if self.mpi is not None: self.mpi.barrier()
#OLD_CPX             MPS_fitting(mps, ref_mps, idMPO, prefit_bond_dims, prefit_nsteps, prefit_noises,
#OLD_CPX                         prefit_conv_tol, 'density_mat', prefit_cutoff, lmpo=idMPO,
#OLD_CPX                         verbose_lvl=self.verbose-1)
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Make the initial MPS complex ====#
#OLD_CPX         cmps = MultiMPS.make_complex(mps, "mps_t")
#OLD_CPX         cmps_t0 = MultiMPS.make_complex(mps, "mps_t0")
#OLD_CPX         if mps.dot != 1: # change to 2dot      #NOTE: Is it for converting to two-site DMRG?
#OLD_CPX             cmps.load_data()
#OLD_CPX             cmps_t0.load_data()
#OLD_CPX             cmps.canonical_form = 'M' + cmps.canonical_form[1:]
#OLD_CPX             cmps_t0.canonical_form = 'M' + cmps_t0.canonical_form[1:]
#OLD_CPX             cmps.dot = 2
#OLD_CPX             cmps_t0.dot = 2
#OLD_CPX             cmps.save_data()
#OLD_CPX             cmps_t0.save_data()
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Initial setups for autocorrelation ====#
#OLD_CPX         idME = MovingEnvironment(idMPO, cmps_t0, cmps, "acorr")
#OLD_CPX             
#OLD_CPX         
#OLD_CPX         #==== Initial setups for time evolution ====#
#OLD_CPX         me = MovingEnvironment(mpo, cmps, cmps, "TE")
#OLD_CPX         self.delayed_contraction = True
#OLD_CPX         if self.delayed_contraction:
#OLD_CPX             me.delayed_contraction = OpNamesSet.normal_ops()
#OLD_CPX         me.cached_contraction = True
#OLD_CPX         me.init_environments()
#OLD_CPX 
#OLD_CPX 
#OLD_CPX         #==== Time evolution ====#
#OLD_CPX         _print('Bond dim in TE : ', mps.info.bond_dim, max_bond_dim)
#OLD_CPX         if mps.info.bond_dim > max_bond_dim:
#OLD_CPX             _print('!!! WARNING !!!')
#OLD_CPX             _print('   The specified max. bond dimension for the time-evolved MPS ' +
#OLD_CPX                    f'({max_bond_dim:d}) is smaller than the max. bond dimension \n' +
#OLD_CPX                    f'  of the initial MPS ({mps.info.bond_dim:d}). This is in general not ' +
#OLD_CPX                    'recommended since the time evolution will always excite \n' +
#OLD_CPX                    '  correlation effects that are absent in the initial MPS.')
#OLD_CPX         if method == TETypes.TangentSpace:
#OLD_CPX             te = TimeEvolution(me, VectorUBond([max_bond_dim]), method)
#OLD_CPX             te.krylov_subspace_size = krylov_size
#OLD_CPX             te.krylov_conv_thrd = krylov_tol
#OLD_CPX         elif method == TETypes.RK4:
#OLD_CPX             te = TimeEvolution(me, VectorUBond([max_bond_dim]), method, n_sub_sweeps_init)
#OLD_CPX         te.cutoff = cutoff                    # for tiny systems, this is important
#OLD_CPX         te.iprint = verbosity
#OLD_CPX         te.normalize_mps = normalize
#OLD_CPX         
#OLD_CPX 
#OLD_CPX         #==== Begin the time evolution ====#
#OLD_CPX         import block2
#OLD_CPX         hamil_te = block2.cpx.su2.HamiltonianQC(
#OLD_CPX             SpinLabel(0), self.n_sites, self.orb_sym, block2.cpx.FCIDUMP())
#OLD_CPX         
#OLD_CPX         if t_sample is not None:
#OLD_CPX             issampled = [False] * len(t_sample)
#OLD_CPX         if save_npy:
#OLD_CPX             np.save('./' + prefix + '.t', ts)
#OLD_CPX             if t_sample is not None: np.save('./'+prefix+'.ts', t_sample)
#OLD_CPX         i_sp = 0
#OLD_CPX         for it, tt in enumerate(ts):
#OLD_CPX 
#OLD_CPX             if self.verbose >= 2:
#OLD_CPX                 _print('\n')
#OLD_CPX                 _print(' Step : ', it)
#OLD_CPX                 _print('>>> TD-PROPAGATION TIME = %10.5f <<<' %tt)
#OLD_CPX             t = time.perf_counter()
#OLD_CPX 
#OLD_CPX             if it != 0: # time zero: no propagation
#OLD_CPX                 dt_ = ts[it] - ts[it-1]
#OLD_CPX                 _print('    DELTA T = %10.5f <<<' % dt_)
#OLD_CPX                 if method == TETypes.RK4:
#OLD_CPX                     te.solve(1, +1j * dt_, cmps.center == 0, tol=exp_tol)
#OLD_CPX                     te.n_sub_sweeps = n_sub_sweeps
#OLD_CPX                 elif method == TETypes.TangentSpace:
#OLD_CPX                     te.solve(2, +1j * dt_ / 2, cmps.center == 0, tol=exp_tol)
#OLD_CPX                     te.n_sub_sweeps = 1                    
#OLD_CPX 
#OLD_CPX             
#OLD_CPX             #==== Autocorrelation and norm ====#
#OLD_CPX             idME.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
#OLD_CPX             acorr = ComplexExpect(idME, max_bond_dim, max_bond_dim)
#OLD_CPX             acorr_t = acorr.solve(False)
#OLD_CPX             if it == 0:
#OLD_CPX                 normsqs = abs(acorr_t)
#OLD_CPX             elif it > 0:
#OLD_CPX                 normsqs = te.normsqs[0]
#OLD_CPX             acorr_t = acorr_t / np.sqrt(normsqs)
#OLD_CPX                             
#OLD_CPX             #==== 2t autocorrelation ====#
#OLD_CPX             if cmps.wfns[0].data.size == 0:
#OLD_CPX                 loaded = True
#OLD_CPX                 cmps.load_tensor(cmps.center)
#OLD_CPX             vec = cmps.wfns[0].data + 1j * cmps.wfns[1].data
#OLD_CPX             acorr_2t = np.vdot(vec.conj(),vec) / normsqs
#OLD_CPX 
#OLD_CPX             #==== Print autocorrelation ====#
#OLD_CPX             if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX                 print('abc ', type(acorr_t), type(acorr_2t))
#OLD_CPX                 ac_print.print_ac(tt, acorr_t, acorr_2t, normsqs)
#OLD_CPX 
#OLD_CPX             if self.mpi is not None: self.mpi.barrier()
#OLD_CPX 
#OLD_CPX             
#OLD_CPX             #==== Stores MPS and/or PDM's at sampling times ====#
#OLD_CPX             if t_sample is not None and np.prod(issampled)==0:
#OLD_CPX                 if it < n_steps-1:
#OLD_CPX                     dt1 = abs( ts[it]   - t_sample[i_sp] )
#OLD_CPX                     dt2 = abs( ts[it+1] - t_sample[i_sp] )
#OLD_CPX                     dd = dt1 < dt2
#OLD_CPX                 else:
#OLD_CPX                     dd = True
#OLD_CPX                     
#OLD_CPX                 if dd and not issampled[i_sp]:
#OLD_CPX                     save_dir = sample_dir + '/mps_sp-' + str(i_sp)
#OLD_CPX                     if self.mpi is not None:
#OLD_CPX                         if self.mpi.rank == 0:
#OLD_CPX                             mkDir(save_dir)
#OLD_CPX                             self.mpi.barrier()
#OLD_CPX                     else:
#OLD_CPX                         mkDir(save_dir)
#OLD_CPX 
#OLD_CPX                     #==== Saving MPS ====##
#OLD_CPX                     if save_mps:
#OLD_CPX                         saveMPStoDir(cmps, save_dir, self.mpi)
#OLD_CPX 
#OLD_CPX                     #==== Calculate 1PDM ====#
#OLD_CPX                     if self.mpi is not None: self.mpi.barrier()
#OLD_CPX                     cmps_cp = cmps.deep_copy('cmps_cp')         # 1)
#OLD_CPX                     if self.mpi is not None: self.mpi.barrier()
#OLD_CPX                     #dm = self.get_one_pdm(True, cmps_cp)
#OLD_CPX                     print('mpi__ = ', self.mpi)
#OLD_CPX                     dm = get_one_pdm(True, 'su2', self.n_sites, hamil_te, cmps_cp,
#OLD_CPX                                      ridx=self.ridx, mpi=self.mpi)
#OLD_CPX                     cmps_cp.info.deallocate()
#OLD_CPX                     dm_full = make_full_dm(self.n_core, dm)
#OLD_CPX                     dm_tr = np.sum( np.trace(dm_full, axis1=1, axis2=2) )
#OLD_CPX                     dm_full = dm_full * nel_t0 / np.abs(dm_tr)      # fm_full is now normalized
#OLD_CPX                     #OLD cmps_cp.deallocate()      # Unnecessary because it must have already been called inside the expect.solve function in the get_one_pdm above
#OLD_CPX                     # NOTE:
#OLD_CPX                     # 1) Copy the current MPS because self.get_one_pdm convert the input
#OLD_CPX                     #    MPS to a real MPS.
#OLD_CPX 
#OLD_CPX                     #==== Save 1PDM ====#
#OLD_CPX                     if save_1pdm: np.save(save_dir+'/1pdm', dm)
#OLD_CPX                         
#OLD_CPX                     #==== Save time info ====#
#OLD_CPX                     self.save_time_info(save_dir, ts[it], it, t_sample[i_sp], i_sp, normsqs, 
#OLD_CPX                                         acorr_t[it], save_mps, save_1pdm, dm)
#OLD_CPX 
#OLD_CPX                     #==== Partial charges ====#
#OLD_CPX                     orbs = np.concatenate((self.core_orbs, self.unordered_site_orbs()),
#OLD_CPX                                           axis=2)
#OLD_CPX                     qmul, qlow = pcharge.calc(self.mol, dm_full, orbs, self.ovl_ao)
#OLD_CPX                     if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX                         q_print.print_pcharge(tt, qlow)
#OLD_CPX                     if self.mpi is not None: self.mpi.barrier()
#OLD_CPX 
#OLD_CPX                     #==== Multipole components ====#
#OLD_CPX                     e_dpole, n_dpole, e_qpole, n_qpole = \
#OLD_CPX                         mpole.calc(self.mol, self.dpole_ao, self.qpole_ao, dm_full, orbs)
#OLD_CPX                     if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX                         mp_print.print_mpole(tt, e_dpole, n_dpole, e_qpole, n_qpole)
#OLD_CPX                     if self.mpi is not None: self.mpi.barrier()
#OLD_CPX 
#OLD_CPX 
#OLD_CPX                     issampled[i_sp] = True
#OLD_CPX                     i_sp += 1
#OLD_CPX 
#OLD_CPX         #==== Print max min imaginary parts (for debugging) ====#
#OLD_CPX         if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX             q_print.footer()
#OLD_CPX 
#OLD_CPX         if self.mpi is None or self.mpi.rank == 0:
#OLD_CPX             mp_print.footer()
#OLD_CPX             
#OLD_CPX     ##############################################################
#OLD_CPX     
#OLD_CPX 
#OLD_CPX     ##############################################################
#OLD_CPX     def __del__(self):
#OLD_CPX         if self.hamil is not None:
#OLD_CPX             self.hamil.deallocate()
#OLD_CPX         if self.fcidump is not None:
#OLD_CPX             self.fcidump.deallocate()
#OLD_CPX         if self.mpo_orig is not None:
#OLD_CPX             self.mpo_orig.deallocate()
#OLD_CPX         release_memory()
#OLD_CPX 
#OLD_CPX ##############################################################




# 1) Does the input ket have to be complex or real?
# 2) Is bond_dims just a list of one element?
# 3) Why is hermitian False by default?




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
