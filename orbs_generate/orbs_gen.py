import numpy as np
from functools import reduce
from pyscf import gto, scf, ao2mo, symm, mcscf
from orbs_generate.local_orbs import localize
from orbs_generate.analyze_orbs import analyze



##########################################################################
def sort_occs(occs, orbs, ergs):
    assert occs is not None, 'sort_occs: The input occs cannot be None.'

    isort = np.argsort(-occs)  
    occs = occs[isort]
    if ergs is not None: ergs = ergs[isort]
    orbs = orbs[:,isort]

    return occs, orbs, ergs
##########################################################################


##########################################################################
def get_hf_orbs():
    pass
##########################################################################


##########################################################################
def get_mcscf_orbs(mol, nCAS, nelCAS, frozen=None, init_mo=None, ss=None, ss_shift=None, 
                   sz=None, wfnsym=None, natorb=False, loc_orb=False, loc_type='PM', 
                   loc_thr=1.9, loc_irrep=True, fcisolver=None):

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
    print('\n>>> Hartree-Fock orbitals analysis <<<')
    analyze(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy)

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
    mc = mcscf.CASSCF(mf, ncas=nCAS, nelecas=nelCAS, frozen=frozen)
    if ss is not None:
        if ss_shift is not None:
            mc.fix_spin_(ss_shift, ss=ss)
        else:
            mc.fix_spin_(ss=ss)

    #==== External FCI solver ====#
    if fcisolver == 'DMRG':
        mc.fcisolver = DMRGCI(mf.mol, maxM=2000, tol=1e-8, num_thrds=1, memory = 7)
        mc.internal_rotation = True
    
        #====   Use the callback function to catch some   ====#
        #==== quantities from inside the iterative solver ====#
        mc.callback = mc.fcisolver.restart_scheduler_()

    if wfnsym is not None:
        mc.fcisolver.wfnsym = wfnsym
    else:
        if mol.groupname == 'c1' or mol.groupname == 'C1':
            mc.fcisolver.wfnsym = 'A'
        
    #==== Run CASSCF ====#
    if init_mo is None:
        mc.kernel(mf.mo_coeff)
    else:
        mc.kernel(init_mo)
    orbs = mc.mo_coeff
    ergs = mc.mo_energy
    ssq, mult = mcscf.spin_square(mc)
    print('Spin square = %-10.6f' % ssq)
    print('Spin multiplicity = %-10.6f' % mult)

    #==== Determine occupations and orbital energies ====#
    rdm_ao = reduce(np.dot, (ovl, mc.make_rdm1(), ovl))    # rdm_ao is in AO rep.    1)
    rdm_mo = reduce(np.dot, (orbs.T, rdm_ao, orbs))        # rdm_mo is in MO rep.
    #OLD o_trans = np.hstack(mol.symm_orb)     # ref: https://pyscf.org/pyscf_api_docs/pyscf.symm.html#pyscf.symm.addons.eigh
    # 1) rdm_mo is in AO rep., needs to transform to an orthogonal rep before feeding
    #    it into symm.eigh.

    #==== If natural orbitals are requested, and determine occs ====#
    if natorb:
        print('>>> Computing natural orbitals <<<')
        #== Get natorb in MO rep. ==#
        cas_sym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mc.mo_coeff)
        natocc, natorb = symm.eigh(rdm_mo, cas_sym)     # 2)
        occs = natocc

        #== Transform back from symm_orb rep. to AO rep. ==#
        orbs = orbs @ natorb
        ergs = None

        # 2) rdm_mo needs to be in an orthonormal basis rep. to be an input to symm.eigh(),
        #    in this case the MO is chosen as the orthonormal basis.
    else:
        occs = np.diag(rdm_mo)

    #== Sort orbs with decreasing occs ==#
    occs, orbs, ergs = sort_occs(occs, orbs, ergs)
        
    #==== Localize orbitals ====#
    if loc_orb:
        print('>>> Performing localization <<<')
        print('Localization type = %s' % loc_type)
        nocc = ncore + nCAS
        
        #== Occupied localized orbitals ==#
        orbs[:,:nocc] = localize(orbs[:,:nocc], mol, loc_type, occs[:nocc], loc_thr,
                                 loc_irrep)

        #== Virtual localized orbitals ==#
        orbs[:,nocc:] = localize(orbs[:,nocc:], mol, loc_type, loc_irrep=loc_irrep)
        for i in range(0, mol.nao):
            occs[i] = np.einsum('j, jk, k', orbs[:,i], rdm_ao, orbs[:,i])
        ergs = None
        occs, orbs, ergs = sort_occs(occs, orbs, ergs)

    #==== Analyze the final orbitals ====#
    print('\n>>> Output orbitals analysis <<<')
    analyze(mol, orbs, occs, ergs)
    
    return orbs
##########################################################################


#=================#
#==== TESTING ====#
#=================#
ss = 2
shift = None
sz = 1

mol = gto.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'cc-pvdz',
    symmetry = 'c2v',
    spin = 2 * sz)

ncore = 2
nelCAS = (mol.nelec[0]-ncore, mol.nelec[1]-ncore)
nCAS = max(nelCAS) + 2

orbs = get_mcscf_orbs(mol, nCAS, nelCAS, ss=ss, ss_shift=shift,
                      sz=(nelCAS[0]-nelCAS[1])/2, natorb=True, loc_orb=True, loc_type='PM',
                      loc_irrep=True)
