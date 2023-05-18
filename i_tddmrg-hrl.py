# HRL: can you adjust the docstrings/comments to our case?

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
DDMRG++ for Green's Function.
using pyscf and block2.

Original version:
     Huanchen Zhai, Nov 5, 2020
Revised: support for mpi
     Huanchen Zhai, Tianyu Zhu Mar 29, 2021
"""

from block2 import SU2, SZ, Global, OpNamesSet, Threading, ThreadingTypes
from block2 import init_memory, release_memory, SiteIndex
from block2 import VectorUInt8, PointGroup, FCIDUMP, QCTypes, SeqTypes, OpNames, Random
from block2 import VectorUBond, VectorDouble, NoiseTypes, DecompositionTypes, EquationTypes
from block2 import OrbitalOrdering, VectorUInt16, TETypes
import time
import numpy as np # HRL: see h2o.py  better import that after all imports are done

# Set spin-adapted or non-spin-adapted here
#SpinLabel = SU2
SpinLabel = SZ

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
from tools import saveMPStoDir, mkDir



        
if hasMPI:
    MPI = MPICommunicator()
else:
    class _MPI:
        rank = 0
    MPI = _MPI()






    

def _print(*args, **kwargs):
    if MPI.rank == 0:
        print(*args, **kwargs)


class MYTDDMRGError(Exception): # HRL: Where do you need this?  >> ANS: Nowhere, it's relic from the orig. code.
    pass


# HRL: Mightbe good to just import this from the gfdmrg code rather than copying it here   >> DONE
def orbital_reorder(h1e, g2e, method='gaopt'):
    """
    Find an optimal ordering of orbitals for DMRG.
    Ref: J. Chem. Phys. 142, 034102 (2015)

    Args:
        method :
            'gaopt' - genetic algorithm, take several seconds
            'fiedler' - very fast, may be slightly worse than 'gaopt'

    Return a index array "midx":
        reordered_orb_sym = original_orb_sym[midx]
    """
    n_sites = h1e.shape[0]
    hmat = np.zeros((n_sites, n_sites))
    xmat = np.zeros((n_sites, n_sites))
    from pyscf import ao2mo
    if not isinstance(h1e, tuple):
        hmat[:] = np.abs(h1e[:])
        g2e = ao2mo.restore(1, g2e, n_sites)
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                xmat[i, j] = abs(g2e[i, j, j, i])
    else:
        assert SpinLabel == SZ
        assert isinstance(h1e, tuple) and len(h1e) == 2
        assert isinstance(g2e, tuple) and len(g2e) == 3
        hmat[:] = 0.5 * np.abs(h1e[0][:]) + 0.5 * np.abs(h1e[1][:])
        g2eaa = ao2mo.restore(1, g2e[0], n_sites)
        g2ebb = ao2mo.restore(1, g2e[1], n_sites)
        g2eab = ao2mo.restore(1, g2e[2], n_sites)
        for i in range(0, n_sites):
            for j in range(0, n_sites):
                xmat[i, j] = 0.25 * abs(g2eaa[i, j, j, i]) \
                    + 0.25 * abs(g2ebb[i, j, j, i]) \
                    + 0.5 * abs(g2eab[i, j, j, i])
    kmat = VectorDouble((np.array(hmat) * 1E-7 + np.array(xmat)).flatten())
    if method == 'gaopt':
        n_tasks = 32
        opts = dict(
            n_generations=10000, n_configs=n_sites * 2,
            n_elite=8, clone_rate=0.1, mutate_rate=0.1
        )
        midx, mf = None, None
        for _ in range(0, n_tasks):
            idx = OrbitalOrdering.ga_opt(n_sites, kmat, **opts)
            f = OrbitalOrdering.evaluate(n_sites, kmat, idx)
            idx = np.array(idx)
            if mf is None or f < mf:
                midx, mf = idx, f
    elif method == 'fiedler':
        idx = OrbitalOrdering.fiedler(n_sites, kmat)
        midx = np.array(idx)
    else:
        midx = np.array(range(n_sites))
    return midx


class MYTDDMRG:
    """
    DDMRG++ for Green's Function for molecules.
    """

    def __init__(self, scratch='./nodex', memory=1 * 1E9, omp_threads=8, verbose=2,
                 print_statistics=True, mpi=None, dctr=True):
        """
        Memory is in bytes.
        verbose = 0 (quiet), 2 (per sweep), 3 (per iteration)
        """

        Random.rand_seed(0)
        # HRL: good to have isize as additional parameter. typically, 200MB is enough.  >> DONE
        #   isize is size of integer stack
        isize = min(int(memory * 0.1), 200000000)
        init_memory(isize=isize, dsize=int(memory - isize), save_dir=scratch)

        Global.threading = Threading(
            ThreadingTypes.OperatorBatchedGEMM | ThreadingTypes.Global, omp_threads, omp_threads, 1)
        Global.threading.seq_type = SeqTypes.Tasked
        Global.frame.load_buffering = False
        Global.frame.save_buffering = False
        Global.frame.use_main_stack = False
        Global.frame.minimal_disk_usage = True

        # HRL: vv MPS_RESTART_DIR is useful for long DMRG runs as it saves the MPS after each sweep. Can you check whethere this is implemnented in tddmrg?
        #   BLOCK2_MPS_DIR saves the MPS in a folder different from scratch. This is useful because in the end the MPS is all you need for restarting/checking your calculation
        # HRL; util.mkDir  == tools.mkDir
        """
    if "BLOCK2_MPS_RESTART_DIR" in os.environ:
        restart_dir = os.environ["BLOCK2_MPS_RESTART_DIR"]
        _print("# SET MPS RESTART DIR TO ",restart_dir)
        util.mkDir(restart_dir)
        Global.frame.restart_dir = restart_dir
    if "BLOCK2_MPS_DIR" in os.environ:
        mps_dir = os.environ["BLOCK2_MPS_DIR"]
        _print("# SET MPS DIR TO ",mps_dir)
        util.mkDir(mps_dir)
        Global.frame.mps_dir = mps_dir
        """
        self.fcidump = None
        self.hamil = None
        self.verbose = verbose
        self.scratch = scratch
        self.mpo_orig = None
        self.print_statistics = print_statistics
        self.mpi = mpi
        self.delayed_contraction = dctr # HRL: change dctr to delayed_contraction  >> DONE
        self.idx = None # reorder
        self.ridx = None # inv reorder

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

    # HRL:  add this here after fcidump set up:
    #           _print("# fcidump symmetrize error:",fcidump.symmetrize(orb_sym))
    #    reason: for  high symmetry, the integrals may be nonzero where they should be zero, due to numerical errors. this routine call sets them to zero. otherwise you get an error. But it is good to monoitor the error so print it

    def init_hamiltonian_fcidump(self, pg, filename, idx=None):
        """Read integrals from FCIDUMP file."""
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        self.fcidump.read(filename)
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(self.fcidump.n_elec, self.fcidump.twos,
                                PointGroup.swap_d2h(self.fcidump.isym))
        self.n_sites = self.fcidump.n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)
        assert pg in ["d2h", "c1"] # HRL generalize!  see usage of swap_pg in block2main and in CAS_example.py
                                    #           PointGroup.swap_d2h needs to be generalized

    def init_hamiltonian(self, pg, n_sites, n_elec, twos, isym, orb_sym,
                         e_core, h1e, g2e, tol=1E-13, idx=None,
                         save_fcidump=None):
        """Initialize integrals using h1e, g2e, etc."""
        assert self.fcidump is None
        self.fcidump = FCIDUMP()
        if not isinstance(h1e, tuple):
            mh1e = np.zeros((n_sites * (n_sites + 1) // 2))
            k = 0
            for i in range(0, n_sites):
                for j in range(0, i + 1):
                    assert abs(h1e[i, j] - h1e[j, i]) < tol
                    mh1e[k] = h1e[i, j]
                    k += 1
            mg2e = g2e.flatten()
            mh1e[np.abs(mh1e) < tol] = 0.0
            mg2e[np.abs(mg2e) < tol] = 0.0
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
            mg2e = tuple(xg2e.flatten() for xg2e in g2e)
            for xmg2e in mg2e:
                xmg2e[np.abs(xmg2e) < tol] = 0.0
            self.fcidump.initialize_sz(
                n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
        self.fcidump.orb_sym = VectorUInt8(orb_sym)
        if idx is not None:
            self.fcidump.reorder(VectorUInt16(idx))
            self.idx = idx
            self.ridx = np.argsort(idx)
        self.orb_sym = VectorUInt8(
            map(PointGroup.swap_d2h, self.fcidump.orb_sym))

        vacuum = SpinLabel(0)
        self.target = SpinLabel(n_elec, twos, PointGroup.swap_d2h(isym))
        self.n_sites = n_sites

        self.hamil = HamiltonianQC(
            vacuum, self.n_sites, self.orb_sym, self.fcidump)

        if save_fcidump is not None:
            if self.mpi is None or self.mpi.rank == 0:
                self.fcidump.orb_sym = VectorUInt8(orb_sym)
                self.fcidump.write(save_fcidump)
            if self.mpi is not None:
                self.mpi.barrier()
        assert pg in ["d2h", "c1"] # HRL generalize! See above

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

    def dmrg(self, bond_dims, noises, n_steps=30, conv_tol=1E-7, cutoff=1E-14, occs=None, bias=1.0):
        """Ground-State DMRG."""

        if self.verbose >= 2:
            _print('>>> START GS-DMRG <<<')
        t = time.perf_counter()

        if self.mpi is not None:
            self.mpi.barrier()

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
        mps = MPS(self.n_sites, 0, 2)
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
        dmrg.davidson_soft_max_iter = 4000
        # HRL: optional low_memory noise: I'd suggest to add this as input 
        #        dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollectedLowMem
        dmrg.noise_type = NoiseTypes.ReducedPerturbative
        dmrg.decomp_type = DecompositionTypes.SVD
        # HRL: I am not sure why this ^^ is used.This here vv should be better. 
        #   dmrg.decomp_type = DecompositionTypes.DensityMatrix     ; IMAM: SVD and DM, aren't they equivalent?
        #   dmrg.noise_type = NoiseTypes.ReducedPerturbativeCollected

        dmrg.iprint = max(self.verbose - 1, 0)
        dmrg.cutoff = cutoff
        dmrg.solve(n_steps, mps.center == 0, conv_tol)

        self.gs_energy = dmrg.energies[-1][0]
        self.bond_dim = bond_dims[-1]

        mps.save_data()
        mps_info.save_data(self.scratch + "/GS_MPS_INFO")
        mps_info.deallocate()

        if self.print_statistics:
            dmain, dseco, imain, iseco = Global.frame.peak_used_memory
            _print("GS PEAK MEM USAGE:",
                   "DMEM = ", MYTDDMRG.fmt_size(dmain + dseco),
                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
                   "IMEM = ", MYTDDMRG.fmt_size(imain + iseco),
                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

        if self.verbose >= 1:
            _print("=== GS Energy = %20.15f" % self.gs_energy)

        if self.verbose >= 2:
            _print('>>> COMPLETE GS-DMRG | Time = %.2f <<<' %
                   (time.perf_counter() - t))

    # one-particle density matrix
    # return value:
    #     pdm[0, :, :] -> <AD_{i,alpha} A_{j,alpha}>
    #     pdm[1, :, :] -> < AD_{i,beta}  A_{j,beta}>
    def get_one_pdm(self, iscomp, mps=None, inmps_name=None):
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
        if iscomp:
            pme.init_environments()
            expect = ComplexExpect(pme, mps.info.bond_dim+100, mps.info.bond_dim+100)   #NOTE
        else:
            pme.init_environments(False)
            expect = Expect(pme, mps.info.bond_dim+100, mps.info.bond_dim+100)   #NOTE
        expect.iprint = max(self.verbose - 1, 0)
        if iscomp:
            expect.solve(False, mps.center == 0)
        else:
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

#   HRL do you need this here vvvv ? Maybe recycle it to allow for optional read-in ; see H2O.py
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


    ##################################################################
    def annihilate(self, bond_dims, cps_bond_dims, cps_noises, cps_conv_tol, cps_n_steps, idxs,
                   addition, cutoff=1E-14, alpha=True, occs=None, bias=1.0, mo_coeff=None,
                   outmps_name='ANN_KET'):
        """Green's function."""
##        ops = [None] * len(idxs)
##        rkets = [None] * len(idxs)
##        rmpos = [None] * len(idxs)

        if self.mpi is not None:
            self.mpi.barrier()

        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            mpo = SimplifiedMPO(mpo, RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo

        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/GS_MPS_INFO")
        mps = MPS(mps_info)
        mps.load_data()

        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO

        if addition: # HRL: remove  >> DONE
            mpo = -1.0 * self.mpo_orig
#need?            mpo.const_e += self.gs_energy
        else:
            mpo = 1.0 * self.mpo_orig
#need?            mpo.const_e -= self.gs_energy

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)

        if self.print_statistics:
            _print('GF MPO BOND DIMS = ', ''.join(
                ["%6d" % (x.m * x.n) for x in mpo.left_operator_names]))
            max_d = max(bond_dims)
            mps_info2 = MPSInfo(self.n_sites, self.hamil.vacuum,
                                self.target, self.hamil.basis)
            mps_info2.set_bond_dimension(max_d)
            _, mem2, disk = mpo.estimate_storage(mps_info2, 2)
            _print("GF EST MAX MPS BOND DIMS = ", ''.join(
                ["%6d" % x.n_states_total for x in mps_info2.left_dims]))
            _print("GF EST PEAK MEM = ", MYTDDMRG.fmt_size(
                mem2), " SCRATCH = ", MYTDDMRG.fmt_size(disk))
            mps_info2.deallocate_mutable()
            mps_info2.deallocate()

#need?        impo = SimplifiedMPO(IdentityMPO(self.hamil),
#need?                             NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
#need?
#need?        if self.mpi is not None:
#need?            impo = ParallelMPO(impo, self.identrule)

#need?        def align_mps_center(ket, ref):
#need?            if self.mpi is not None:
#need?                self.mpi.barrier()
#need?            cf = ket.canonical_form
#need?            if ref.center == 0:
#need?                ket.center += 1
#need?                ket.canonical_form = ket.canonical_form[:-1] + 'S'
#need?                while ket.center != 0:
#need?                    ket.move_left(mpo.tf.opf.cg, self.prule)
#need?            else:
#need?                ket.canonical_form = 'K' + ket.canonical_form[1:]
#need?                while ket.center != ket.n_sites - 1:
#need?                    ket.move_right(mpo.tf.opf.cg, self.prule)
#need?                ket.center -= 1
#need?            if self.verbose >= 2:
#need?                _print('CF = %s --> %s' % (cf, ket.canonical_form))


#NOTE: check if ridx is not none
#NOTE: change dctr to delayed____ (DONE)


# HRL: This is not yet generalized / adapted for our case; remember our discussion

        if mo_coeff is None:
            if self.ridx is not None:
                gidxs = self.ridx[np.array(idxs)]
            else:
                gidxs = idxs
        else:
            if self.idx is not None:
                mo_coeff = mo_coeff[:, self.idx]
            gidxs = list(range(self.n_sites))
##            ops = [None] * self.n_sites
            _print('idxs = ', idxs, 'gidxs = ', gidxs)

#        for ii, idx in enumerate(gidxs):
#        idx = idxs  (IT WAS THIS LINE WHICH WAS THE PROBLEM)
        idx = self.ridx[idxs]
        if SpinLabel == SZ:
            if addition:
                ops = OpElement(OpNames.C, SiteIndex(
                    (idx, ), (0 if alpha else 1, )), SZ(1, 1 if alpha else -1, self.orb_sym[idx]))
            else:
                ops = OpElement(OpNames.D, SiteIndex(
                    (idx, ), (0 if alpha else 1, )), SZ(-1, -1 if alpha else 1, self.orb_sym[idx]))
        else:
            if addition:
                ops = OpElement(OpNames.C, SiteIndex(
                    (idx, ), ()), SU2(1, 1, self.orb_sym[idx]))
            else:
                ops = OpElement(OpNames.D, SiteIndex(
                    (idx, ), ()), SU2(-1, 1, self.orb_sym[idx]))

#        for ii, idx in enumerate(idxs):
        if self.mpi is not None:
            self.mpi.barrier()
        if self.verbose >= 2:
            _print('>>> START Compression Site = %4d <<<' % idx)
        t = time.perf_counter()

        rket_info = MPSInfo(self.n_sites, self.hamil.vacuum,
                            self.target + ops.q_label, self.hamil.basis)
        rket_info.tag = 'DKET%d' % idx
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



        #==== IMAM ====#
        rket_info.save_data(self.scratch + "/" + outmps_name)


        
        rkets = MPS(self.n_sites, mps.center, 2)
        rkets.initialize(rket_info)
        rkets.random_canonicalize()
        
        rkets.save_mutable()
        rkets.deallocate()
        rket_info.save_mutable()
        rket_info.deallocate_mutable()

        if mo_coeff is None:
            # the mpo and gf are in the same basis
            # the mpo is SiteMPO
            rmpos = SimplifiedMPO(
                SiteMPO(self.hamil, ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))
        else:
            # the mpo is in mo basis and gf is in ao basis
            # the mpo is sum of SiteMPO (LocalMPO)
            ao_ops = VectorOpElement([None] * self.n_sites)
            for ix in range(self.n_sites):
                ao_ops[ix] = ops[ix] * mo_coeff[idx, ix]
            rmpos = SimplifiedMPO(
                LocalMPO(self.hamil, ao_ops), NoTransposeRule(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))

        if self.mpi is not None:
            rmpos = ParallelMPO(rmpos, self.siterule)

        if len(cps_noises) == 1 and cps_noises[0] == 0:
            pme = None
        else:
            pme = MovingEnvironment(mpo, rkets, rkets, "PERT")
            pme.init_environments(False)
        rme = MovingEnvironment(rmpos, rkets, mps, "RHS")
        rme.init_environments(False)
        if self.delayed_contraction:
            if pme is not None:
                pme.delayed_contraction = OpNamesSet.normal_ops()
            rme.delayed_contraction = OpNamesSet.normal_ops()

        cps = Linear(pme, rme, VectorUBond(cps_bond_dims),
                     VectorUBond([mps.info.bond_dim+100]), VectorDouble(cps_noises))
        # HRL: see DMRG regarding noise/decomp type
        cps.noise_type = NoiseTypes.ReducedPerturbative
        cps.decomp_type = DecompositionTypes.SVD
        if pme is not None:
            cps.eq_type = EquationTypes.PerturbativeCompression
        cps.iprint = max(self.verbose - 1, 0)
        cps.cutoff = cutoff
        cps.solve(cps_n_steps, mps.center == 0, cps_conv_tol)
        
        if self.verbose >= 2:
            _print('>>> COMPLETE Compression Site = %4d | Time = %.2f <<<' %
                   (idx, time.perf_counter() - t))
    ##################################################################


    ##################################################################
    def time_propagate(self, inmps_name, bond_dim: int, tmax: float, dt: float, n_sub_sweeps=2,
                       n_sub_sweeps_init=4, exp_tol=1e-6, t_sample=None, print_mps=False,
                       print_1pdm=False, print_2pdm=False):
        # HRL: document input

        #==== Prepare Hamiltonian MPO ====#
        if self.mpi is not None:
            self.mpi.barrier()

        if self.mpo_orig is None:
            mpo = MPOQC(self.hamil, QCTypes.Conventional)
            # HRL: no AncillaMPO...    >> DONE
            mpo = SimplifiedMPO(AncillaMPO(mpo), RuleQC(), True, True,
                                OpNamesSet((OpNames.R, OpNames.RD)))
            self.mpo_orig = mpo
            
        if self.mpi is not None:
            if SpinLabel == SU2:
                from block2.su2 import ParallelMPO
            else:
                from block2.sz import ParallelMPO
        mpo = 1.0 * self.mpo_orig

        #mpo.const_e = 0 # hrl: sometimes const_e causes trouble for AncillaMPO
#need?        mpo = IdentityAddedMPO(mpo) # hrl: alternative

        if self.mpi is not None:
            mpo = ParallelMPO(mpo, self.prule)


        #==== Load the initial MPS ====#
        mps_info = MPSInfo(0)
        mps_info.load_data(self.scratch + "/" + inmps_name)
        mps = MPS(mps_info)
        mps.load_data()
        mps.info.load_mutable()
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
        idMPO = SimplifiedMPO(IdentityMPO(self.hamil), RuleQC(), True, True)
        if self.mpi is not None:
            idMPO = ParallelMPO(idMPO, self.identrule)
        idME = MovingEnvironment(idMPO, cmps_t0, cmps, "acorr")
##        idME.init_environments()
##        acorr = ComplexExpect(idME, bond_dim, bond_dim)

        
        #==== Initial setups for time evolution ====#
        me = MovingEnvironment(mpo, cmps, cmps, "TE")
        self.delayed_contraction = True
        if self.delayed_contraction:
            me.delayed_contraction = OpNamesSet.normal_ops()
        me.cached_contraction = True
        me.init_environments()


        #==== Time evolution ====#
        # HRL: generalize this and make it input vvv   >> DONE
        method = TETypes.RK4
##        method = TETypes.TangentSpace
        te = TimeEvolution(me, VectorUBond([bond_dim]), method, n_sub_sweeps_init)
        # HRL: vvv input with default 0   >> DONE
        te.cutoff = 0  # for tiny systems, this is important
        # HRL: vvv input  >> DONE
        te.iprint = 6  # ft.verbose
        te.normalize_mps = False

        n_steps = int(tmax/dt + 1)
        ts = np.linspace(0, tmax, n_steps) # times
        issampled = [False] * len(t_sample)
        i_sp = 0
        for it, tt in enumerate(ts):

            if self.verbose >= 2:
                _print('\n')
                _print(' Step : ', it)
                _print('>>> TD-PROPAGATION TIME = %10.5f <<<' %tt)
            t = time.perf_counter()

            if it != 0: # time zero: no propagation
                if method == TETypes.RK4:
                    te.solve(1, +1j * dt, cmps.center == 0, tol=exp_tol)
                elif method == TETypes.TangentSpace:
                    te.solve(2, +1j * dt / 2, cmps.center == 0, tol=exp_tol)
                te.n_sub_sweeps = n_sub_sweeps

                #==== Autocorrelation ====#
                # HRL: If normalize_mps==False, you need to take the decaying norm into account for all observables
                #       simplest way is to use ComplexExpect 

                idME.init_environments()   # NOTE: Why does it have to be here instead of between 'idMe =' and 'acorr =' lines.
                # HRL ^^^ is needed whenever MPS is changed. This does the first contraction with the MPS
                acorr = ComplexExpect(idME, bond_dim, bond_dim)
                acorr_t = acorr.solve(False) * -1j
                _print('acorr_t, abs = ', acorr_t, abs(acorr_t))
                # HRL: as discussed, add double time trick

                # HRL: I think one-PDM should not be too expensive. I would do it at every time step
                #       also, you need to get this for t=0

                #==== Stores MPS and/or PDM's at sampling times ====#
                if t_sample is not None and np.prod(issampled)==0:
                    # HRL: vv what do you do here?
                    #           Looks very complicated.
                    #           If you just want to decide when to sample,
                    #               samplePoints = set(range(0,len(ts), t_sample))
                    #               and then "if it in samplePoints"...
                    #       btw, "t_sample" means number of samples? Why not call it then "n_sample"?
                    #            t_sample sounds to me like an array of sampling times
                    if it < n_steps-1:
                        dt1 = abs( ts[it]   - t_sample[i_sp] )
                        dt2 = abs( ts[it+1] - t_sample[i_sp] )
                        dd = dt1 < dt2
                    else:
                        dd = True
                    _print('i_sp, dt1, dt2 = ', i_sp, dt1, dt2)
                    if dd and not issampled[i_sp]:
                        # HRL: self.scratch is typically deletedafter the run. better save it somewhere else. see also above re. BLOCK2_MPS_DIR
                        sample_dir = self.scratch + '/mps_sp-' + str(i_sp)
                        mkDir(sample_dir)
     
                        ##### Using saveMPStoDir #####
                        if print_mps:# HRL: you don't print here   >> DONE
                            saveMPStoDir(cmps, sample_dir, self.mpi)
                        #####

                    
                    ##### Using deep_copy and save_data's #####
#                    if MPI is not None: MPI.barrier()
#                    mps_sp = cmps.deep_copy('mps_sp.'+str(i_sp))
#                    if MPI is not None: MPI.barrier()                    
#they cause segfault and other error                    mps_sp.save_data()
#they cause segfault and other error                    mps_sp.save_mutable()
#they cause segfault and other error                    mps_sp.deallocate()
#they cause segfault and other error                    mps_sp.info.save_mutable()
#they cause segfault and other error                    mps_sp.info.deallocate_mutable()
#                    mps_sp.info.save_data(self.scratch + '/mps_sp_info.' + str(i_sp))
#                    mps_sp.info.deallocate()
                    #####

                        if print_1pdm:
                            dm = self.get_one_pdm(True, cmps)
                            np.save(sample_dir+'/1pdm', dm)
                            _print('DM a = ', dm[0,:,:]) # HRL I would not do that.  or only for tiny systems
                        
                        issampled[i_sp] = True
                        i_sp += 1








                        
#need?        mps.save_data()
#need?        mps_info.deallocate()


##        if self.print_statistics:
##            _print("GF PEAK MEM USAGE:",
##                   "DMEM = ", MYTDDMRG.fmt_size(dmain + dseco),
##                   "(%.0f%%)" % (dmain * 100 / (dmain + dseco)),
##                   "IMEM = ", MYTDDMRG.fmt_size(imain + iseco),
##                   "(%.0f%%)" % (imain * 100 / (imain + iseco)))

##        return gf_mat

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
