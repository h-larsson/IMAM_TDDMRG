import os
import sys

def printDummyFunction(*args, **kwargs):
    """ Does nothing"""
    pass

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


try:
    from block2.sz import MPICommunicator
    hasMPI = True
    MPI = MPICommunicator()
    _print = getVerbosePrinter(MPI.rank==0,flush=True)
except ImportError:
    MPICommunicator = None
    hasMPI = False
    MPI = None
    _print = getVerbosePrinter(True,flush=True)
from pyscf import gto, scf, tools
from pyscf import mcscf, symm, ao2mo
from pyscf.mcscf import casci_symm
if hasMPI:
    from mpi4py import MPI as MPIpy
    commPy = MPIpy.COMM_WORLD
    #print(MPI.rank,"rank",flush=True)
    assert commPy.Get_rank() == MPI.rank, f"{commPy.Get_rank()} vs {MPI.rank}"
    assert commPy.Get_size() == MPI.size, f"{commPy.Get_size()} vs {MPI.size}"
    MAIN_PROCESS = MPI.rank == 0
else:
    MAIN_PROCESS = True

def getFCIDUMP(file, mol, wfnSym, h1e, eri, nElec: Tuple[int], eCore, orbsym, tol):
    from block2 import FCIDUMP
    assert hasattr(nElec, "__len__"), "nElec needs to be nela,nelb tuple"
    assert len(nElec) == 2, "nElec needs to be nela,nelb tuple"
    nTot = h1e.shape[0]
    assert np.all(np.array(orbsym) > 0), f"molpro orbsym needed!{orbsym}"
    assert len(orbsym) == nTot, f"nTot={nTot} vs |orbsym|={len(orbsym)}"
    if file is not None:
        tools.fcidump.from_integrals(file, h1e, eri, nTot, nElec, eCore,
                                     orbsym=orbsym, tol=tol)
        fcidump = FCIDUMP()
        fcidump.read(file)
    else:
        eri = ao2mo.restore(8, eri, nTot)
        h1e = h1e[np.tril_indices_from(h1e)].ravel()
        h1e[abs(h1e) < tol] = 0.0
        eri = eri.ravel()
        eri[abs(eri) < tol] = 0.0
        wfnSym_molpro = tools.fcidump.ORBSYM_MAP[mol.groupname][wfnSym]
        fcidump = FCIDUMP()
        nela, nelb = nElec
        fcidump.initialize_su2(nTot, nela + nelb, abs(nela - nelb), wfnSym_molpro, eCore, h1e, eri)
        fcidump.orb_sym = VectorUInt8(orbsym)
    return fcidump


USE_FCIDUMP = False
FCIDUMP_TOL = 1e-12

nCas = 12
nElCas = 12
nCore = 10 # THESE WILL BE REMOVED (+ nAO - nCas - nCore virtual orbitals)

mol = gto.Mole()
mol.verbose = 2
mol.atom = [
    ['Cr', (0.000000, 0.000000, -1 / 2)],
    ['Cr', (0.000000, 0.000000, 1 / 2)],
]
mol.basis = 'cc-pwCVTZ-DK'
mol.symmetry = True
mol.max_memory = 8000
mol.symmetry_subgroup = "D2h" 
mol.build()
if MAIN_PROCESS:
    mf = scf.RHF(mol)

    MO_COEFF = np.load("mo_coeff.npy") # HRL often it is good to compute and save the mo_coeffs once and then reuse it for various runs
    _mcCI = mcscf.CASCI(mf, ncas=mol.nao-nCore, nelecas=nElCas , ncore=nCore)
    _mcCI.mo_coeff = MO_COEFF
    _mcCI.mo_coeff = casci_symm.label_symmetry_(_mcCI, MO_COEFF)
    wfnSym = _mcCI.fcisolver.wfnsym
    wfnSym = wfnSym if wfnSym is not None else 0
    h1e, eCore = _mcCI.get_h1cas()
    orbSym = np.array(_mcCI.mo_coeff.orbsym)[nCore:]
    nTot = h1e.shape[0]
    nExt = nTot - nCas - nThawed
    _print(f"# nTot= {nTot}, nCas={nCas}, nExt={nExt}")
    assert nTot == mol.nao - nCore
    _print("# groupName=",mol.groupname)
    _print("# orbSym=",orbSym)
    _print("# orbSym=",[symm.irrep_id2name(mol.groupname,s) for s in orbSym])
    _print("# wfnSym=",wfnSym, symm.irrep_id2name(mol.groupname,wfnSym),flush=True)
    assert wfnSym == 0, "Want A1g state"
    molpro_orbsym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbSym]
    assert orbSym.size == nTot

    eri = _mcCI.get_h2cas()
    eri = np.require(eri,dtype=np.float64)
    h1e = np.require(h1e,dtype=np.float64)
    eriShape = eri.shape
    h1eShape = h1e.shape
    nelecas = _mcCI.nelecas
    del _mcCI,mf
else:
    eri = h1e = eCore = molpro_orbsym = nelecas = wfnSym = None
    E_CASSCF = None
    eriShape = h1eShape = None
    nExt = 0

if hasMPI:
    eCore = commPy.bcast(eCore, root = 0)
    molpro_orbsym = commPy.bcast(molpro_orbsym, root = 0)
    wfnSym = commPy.bcast(wfnSym, root = 0)
    nExt = commPy.bcast(nExt, root = 0)
    eriShape = commPy.bcast(eriShape, root = 0)
    h1eShape = commPy.bcast(h1eShape, root = 0)
    E_CASSCF = commPy.bcast(E_CASSCF, root = 0)
    nelecas = commPy.bcast(nelecas, root = 0)
    if not MAIN_PROCESS:
        eri = np.empty(eriShape,dtype=np.float64)
        h1e = np.empty(h1eShape,dtype=np.float64)
    commPy.Bcast([eri, MPIpy.DOUBLE], root=0)
    commPy.Bcast([h1e, MPIpy.DOUBLE], root=0)

fcidump = getFCIDUMP(FCIDUMP_FILE if USE_FCIDUMP else None,
                     mol, wfnSym, h1e, eri, nelecas, eCore, molpro_orbsym, FCIDUMP_TOL)
pg = mol.groupname.lower()
del mol
del h1e, eri

# Hamiltonian initialization:
# HRL vvv see comment in i_tddmrg on "assert pg ..."
swap_pg = getattr(PointGroup, "swap_" + pg)
orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
if MAIN_PROCESS:
    assert np.allclose(np.array(orb_sym) , orbSym), f"{orb_sym} vs {orbSym}"
target = SZ(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))
#...
