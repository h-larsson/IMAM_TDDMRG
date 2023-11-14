import numpy as np
from functools import reduce
from pyscf import gto, scf, dft, ao2mo, symm, mcscf
from pyscf.dmrgscf import DMRGCI
from IMAM_TDDMRG.utils.util_print import print_matrix
from util_orbs import sort_orbs


##########################################################################
def get_rhf_orbs(mol, sz=None, save_rdm=True, natorb=False): 
    '''
    Input parameters:
    ----------------

    mol = Mole object.
    sz = The z-projection of the total spin operator (S).

    Outputs:
    -------
 
    The output is a dictionary.
    '''

    if natorb:
        print('>>> ATTENTION <<<')
        print('Natural orbitals are specified for get_rhf_orbs which does not have any ' + \
              'effect because in Hartree-Fock method, the natural orbitals are the ' + \
              'same as the canonical orbitals.')
    
    #==== Set up system (mol) ====#
    dnel = mol.nelec[0] - mol.nelec[1]
    if sz is not None:
        mol.spin = int(2 * sz)
        assert mol.spin == dnel, 'get_rhf_orbs:' + \
            f'The chosen value of sz ({sz}) is inconsistent with the difference ' + \
            f'between the number of alpha and beta electrons ({dnel}).'
    else:
        mol.spin = dnel
    mol.build()
    
    #==== Run HF ====#
    print('\n\n')
    print('==================================')
    print('>>>> HARTREE-FOCK CALCULATION <<<<')
    print('==================================')
    print('')
    print('No. of MO / no. of electrons = %d / (%d, %d)' % 
          (mol.nao, mol.nelec[0], mol.nelec[1]))
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-7
    mf.kernel()
    orbs, occs, ergs = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    ssq, mult = mf.spin_square()
    print('Spin square = %-10.6f' % ssq)
    print('Spin multiplicity = %-10.6f' % mult)

    orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'erg', 'as')

    
    #==== What to output ====#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    outs['ergs'] = ergs
    if save_rdm: outs['rdm'] = np.diag(occs)
    
    return outs
##########################################################################


##########################################################################
def get_casscf_orbs(mol, nCAS, nelCAS, init_mo, frozen=None, ss=None, ss_shift=None, 
                    sz=None, wfnsym=None, natorb=False, init_basis=None, sort_out=None, 
                    save_rdm=True, verbose=2, fcisolver=None, maxM=None, sweep_tol=1.0E-7,
                    dmrg_nthreads=1):
    '''
    Input parameters:
    ----------------

    mol = Mole object of the molecule.
    nCAS = The number of active space orbitals.
    nelCAS = The number of electrons in the active space. The numbers of electrons in CAS
             and in the frozen orbitals (see frozen below) do not have to add up to the total
             number of electrons. The difference will be treated as the core electrons. That
             means there will be ncore/2 orbitals that are doubly occupied throughout the 
             SCF iteration, where ncore is the number of core electrons.
    frozen = Orbitals to be frozen, i.e. not optimized. If given an integer, it is used 
             as the number of the lowest orbitals to be frozen. If it is given a list of
             integers, it must contain the base-1 indices of the orbitals to be frozen.
    init_mo = The guess orbitals to initiate the CASSCF iteration.
    ss = The value of <S^2> = S(S+1) of the desired state. Note that the value of <S^2>
         given to this function is not guaranteed to be achieved at the end of the CASSCF
         procedure. Adjust ss_shift if the final <S^2> is not equal to the value of ss.
    ss_shift = A parameter to control the energy shift of the CASSCF solution when ss is
               not None.
    sz = The z-projection of the total spin operator (S).
    wfnsym = The irreducible representation of the desired multi-electron state.
    natorb = If True, the natural orbitals will be returned, otherwise, the 
             canonical orbitals will be returned.
    save_rdm = Return the 1RDM in the MO rep.
    fcisolver = Controls the use of external FCI solver, at the moment only 'DMRG' is
                is supported. By default, it will use the original CASSCF solver.
    maxM = The maximum bond dimension for the DMRG solver. Only meaningful if fcisolver =
           'DMRG'.

    Outputs:
    -------
 
    The output is a dictionary.
    '''

    
    #==== Set up system (mol) ====#
    assert isinstance(nelCAS, tuple)
    dnel = nelCAS[0] - nelCAS[1]
    if sz is not None:
        mol.spin = int(2 * sz)
        assert mol.spin == dnel, \
            f'The chosen value of sz ({sz}) is inconsistent with the difference ' + \
            f'between the number of alpha and beta electrons ({dnel}).'
    else:
        mol.spin = dnel
    mol.build()
    ovl = mol.intor('int1e_ovlp')

    #==== Determine the number of core orbitals ====#
    nelcore = [None] * 2
    for i in range (0,2): nelcore[i] = mol.nelec[i] - nelCAS[i]
    assert nelcore[0] == nelcore[1], 'The numbers of core alpha and beta electrons ' + \
        f'do not match. n_core[alpha] = {nelcore[0]} vs. n_core[beta] = {nelcore[1]}.'
    ncore = nelcore[0]
    
    #==== Run HF because it is needed by CASSCF ====#
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-7
    mf.kernel()

    #==== Set up the CAS ====#
    print('\n\n')
    print('============================')
    print('>>>> CASSCF CALCULATION <<<<')
    print('============================')
    print('')
    print('No. of MO / no. of electrons = %d / (%d, %d)' % 
          (mol.nao, mol.nelec[0], mol.nelec[1]))
    print('No. of core orbitals / no. of core electrons = %d / (%d, %d)' %
          (ncore, nelcore[0], nelcore[1]))
    print('No. of CAS orbitals / no. of CAS electrons = %d / (%d, %d)' %
          (nCAS, nelCAS[0], nelCAS[1]))
    print('Frozen orbitals = ', end='')
    if isinstance(frozen, int):
        for i in range(0, frozen): print(' %d' % (i+1), end='')
    elif isinstance(frozen, list):
        for i in frozen: print(' %d' % frozen[i], end='')
    elif frozen is None:
        print('no orbitals frozen', end='')
    print('')
    
    mc = mcscf.CASSCF(mf, ncas=nCAS, nelecas=nelCAS, frozen=frozen)
    if ss is not None:
        if ss_shift is not None:
            mc.fix_spin_(ss_shift, ss=ss)
        else:
            mc.fix_spin_(ss=ss)

    #==== External FCI solver ====#
    if fcisolver == 'DMRG':
        if maxM is None:
            raise ValueError('get_casscf_orbs: maxM is needed when fcisolver = ' + \
                             '\'DMRG\'.')
        mc.fcisolver = DMRGCI(mf.mol, maxM=maxM, tol=sweep_tol, num_thrds=dmrg_nthreads,
                              memory = 7)
        mc.internal_rotation = True
    
        #====   Use the callback function to catch some   ====#
        #==== quantities from inside the iterative solver ====#
        mc.callback = mc.fcisolver.restart_scheduler_()

    if wfnsym is not None:
        mc.fcisolver.wfnsym = wfnsym
    else:
        if mol.groupname == 'c1' or mol.groupname == 'C1':
            mc.fcisolver.wfnsym = 'A'
        else:
            pass
    
    #==== Run CASSCF ====#
    if init_basis is not None:
        mol0 = mol.copy()
        mol0.basis = init_basis
        mol0.build()
        init_mo0 = mcscf.project_init_guess(mc, init_mo, prev_mol=mol0)
    else:
        init_mo0 = init_mo.copy()
    mc.kernel(init_mo0)
    orbs = mc.mo_coeff
    ergs = mc.mo_energy
    if fcisolver is None:
        ssq, mult = mcscf.spin_square(mc)
        print('Spin square = %-10.6f' % ssq)
        print('Spin multiplicity = %-10.6f' % mult)

    #==== Determine occupations and orbital energies ====#
    rdm_ao = reduce(np.dot, (ovl, mc.make_rdm1(), ovl))    # rdm_ao is in AO rep.    1)
    rdm_mo = reduce(np.dot, (orbs.T, rdm_ao, orbs))        # rdm_mo is in MO rep.
    #OLD o_trans = np.hstack(mol.symm_orb)     # ref: https://pyscf.org/pyscf_api_docs/pyscf.symm.html#pyscf.symm.addons.eigh
    # 1) rdm_ao is in AO rep., needs to transform to an orthogonal rep before feeding
    #    it into symm.eigh below.

    #==== If natural orbitals are requested, and determine occs ====#
    if natorb:
        print('>>> Computing natural orbitals <<<')
        #== Get natorb in MO rep. ==#
        cas_sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff)
        natocc, natorb = symm.eigh(rdm_mo, cas_sym)     # 2)
        occs = natocc

        #== Transform back from symm_orb rep. to AO rep. ==#
        #WARNING: Check if the line below is correct!
        orbs = orbs @ natorb
        ergs = None

        if sort_out is not None:
            assert sort_out[0] == 'occ', \
                'At the moment sorting using energies when orbital source is ' + \
                'CASSCF is not supported.'
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, sort_out[0], sort_out[1])
        else:
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'occ', 'de')
        rdm_mo = np.diag(occs)
        # 2) rdm_mo needs to be in an orthonormal basis rep. to be an input to symm.eigh(),
        #    in this case the MO is chosen as the orthonormal basis.
    else:
        occs = np.diag(rdm_mo).copy()
        if sort_out is not None:
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, sort_out[0], sort_out[1])
        else:
            orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'occ', 'de')
        rdm_mo = reduce(np.dot, (orbs.T, rdm_ao, orbs))     # Take into account the reordering of the columns of orbs.


    #==== Analyze ====#
    print('\n')
    print('=====================================')
    print('*** Analysis of the CASSCF result ***')
    print('=====================================')
    print('')
    cc = np.einsum('im, mn, nj -> ij', init_mo0.T, ovl, orbs)
    print('Overlap between the final and initial (guess) orbitals ' +
          '(row -> initial, column -> final):')
    print_matrix(cc)
    print('')
    mc.verbose = verbose
    mc.analyze()
        
        
    #==== What to output ====#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    if ergs is not None: outs['ergs'] = ergs
    if save_rdm: outs['rdm'] = rdm_mo

    return outs
##########################################################################


##########################################################################
def get_dft_orbs(mol, xc, sz=None, save_rdm=True, natorb=False):
    '''
    Input parameters:
    ----------------

    mol = Mole object.
    sz = The z-projection of the total spin operator (S).

    Outputs:
    -------
 
    The output is a dictionary.
    '''

    if natorb:
        print('>>> ATTENTION <<<')
        print('Natural orbitals are specified for get_dft_orbs which does not have any ' + \
              'effect because in DFT, the natural orbitals are the same as the canonical ' + \
              'orbitals.')

        
    #==== Set up system (mol) ====#
    print('\n\n')
    print('=========================')
    print('>>>> DFT CALCULATION <<<<')
    print('=========================')
    print('')
    print('No. of MO / no. of electrons = %d / (%d, %d)' % 
          (mol.nao, mol.nelec[0], mol.nelec[1]))
    dnel = mol.nelec[0] - mol.nelec[1]
    if sz is not None:
        mol.spin = int(2 * sz)
        assert mol.spin == dnel, 'get_dft_orbs:' + \
            f'The chosen value of sz ({sz}) is inconsistent with the difference ' + \
            f'between the number of alpha and beta electrons ({dnel}).'
    else:
        mol.spin = dnel
    mol.build()


    #==== Run DFT ====#
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.kernel()
    orbs, occs, ergs = mf.mo_coeff, mf.mo_occ, mf.mo_energy
    ssq, mult = mf.spin_square()
    print('Spin square = %-10.6f' % ssq)
    print('Spin multiplicity = %-10.6f' % mult)

    orbs, occs, ergs = sort_orbs(orbs, occs, ergs, 'erg', 'as')


    #==== What to output ====#
    outs = {}
    outs['orbs'] = orbs
    outs['occs'] = occs
    outs['ergs'] = ergs
    if save_rdm: outs['rdm'] = np.diag(occs)

    return outs
##########################################################################





#=================#
#==== TESTING ====#
#=================#
if __name__ == "__main__":
    #ss = 2
    #shift = None
    #sz = 1
    #
    #mol = gto.M(
    #    atom = 'O 0 0 0; O 0 0 1.2',
    #    basis = 'cc-pvdz',
    #    symmetry = True,
    #    symmetry_subgroup = 'c2v',
    #    spin = 2 * sz)
    #
    #ncore = 2
    #nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
    #nCAS = max(nelCAS) + 2
    
    
    
    
    ss = 0
    shift = None
    sz = 0
    
    mol = gto.M(
        atom = 'C 0 0 -0.6; C 0 0 0.6',
        basis = 'cc-pvdz',
        symmetry = True,
        symmetry_subgroup = 'c2v',
        spin = 2 * sz)
    
    ncore = 2
    nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
    #nCAS = max(nelCAS) + 8
    nCAS = max(nelCAS) + 3
    
    
    
    #ss = 0
    #shift = None
    #sz = 0
    #
    #mol = gto.M(
    #    atom = 'H 0 0 0; H 0 0 1.2',
    #    basis = 'cc-pvdz',
    #    symmetry = 'd2h',
    #    spin = 2 * sz)
    #
    #ncore = 0
    #nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
    #nCAS = max(nelCAS) + 5
    
    
    print('\n\n\n!!! CASSCF !!!')
    orbs = get_casscf_orbs(mol, nCAS, nelCAS, ss=ss, ss_shift=shift,
                          sz=sz, natorb=False, loc_orb=True, loc_type='PM',
                           loc_irrep=True, fcisolver=None)
    
    
    print('\n\n\n!!! CASSCF with DMRG !!!')
    orbs = get_casscf_orbs(mol, nCAS, nelCAS, ss=ss, ss_shift=shift,
                          sz=sz, natorb=False, loc_orb=False, loc_type='PM',
                           loc_irrep=True, fcisolver='DMRG', maxM=400)
    
    
    print('\n\n\n!!! RHF !!!')
    orbs = get_rhf_orbs(mol, sz, loc_orb=False, loc_type='PM', loc_irrep=True)
