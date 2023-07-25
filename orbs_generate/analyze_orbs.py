import numpy as np
from pyscf import gto, symm
from scipy.linalg import eigh


##########################################################################
def analyze(mol, ocoeff, oocc=None, oerg=None):
    n_mo = mol.nao
    
    #==== Determine the no. of unique spin channels ====#
    if len(ocoeff.shape) == 3:
        ns = 2
    elif len(ocoeff.shape) == 2:
        ns = 1

    #==== Orbital occupations ====#
    if oocc is not None:
        oocc_ = np.zeros((ns, n_mo))
        if ns == 1:
            oocc_[0,:] = oocc
        elif ns == 2:
            oocc_ = oocc

    #==== Orbital energies ====#
    if oerg is not None:
        oerg_ = np.zeros((ns, n_mo))
        if ns == 1:
            oerg_[0,:] = oerg
        elif ns == 2:
            oerg_ = oerg            
        
    #==== Recast MO coefficients into another array ====#
    mo_c = np.zeros((ns, n_mo, n_mo))
    if ns == 1:
        mo_c[0,:,:] = ocoeff
    elif ns == 2:
        mo_c = ocoeff

    #==== Orbital symmetry ====#
    osym_l = [None] * ns
    osym = [None] * ns
    for i in range(0, ns):
        try:
            osym_l[i] = list(symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb,
                                                 mo_c[i,:,:]))
            osym[i] = [symm.irrep_name2id(mol.groupname, s) for s in osym_l[i]]
            issym = True
            of = 0
        except ValueError:
            osym_l[i] = ['UNSYM'] * mo_c.shape[2]
            osym[i] = [-1] * mo_c.shape[2]
            issym = False
            of = 3
        
    #==== Get the index of the sorted MO coefficients ====#
    idsort = np.zeros((ns, n_mo, n_mo))
    for s in range(0, ns):
        for i in range(0, n_mo):
            idsort[s,:,i] = np.argsort(np.abs(mo_c[s,:,i]))   # Sort from smallest to largest.
            idsort[s,:,i] = idsort[s,::-1,i]                  # Sort from largest to smallest.
    
    #==== Construct various labels ====#
    atom_label = [None] * n_mo
    sph_label =  [None] * n_mo
    for i in range(0, n_mo):
        ao_label = mol.ao_labels(fmt=False)[i]
        atom_label[i] = ao_label[1] + str(ao_label[0])
        if ao_label[3] == '':
            sph_label[i] = 's'
        else:
            sph_label[i] = ao_label[3]
    
    #==== Print orbital properties ====#
    ln_atom = [len(ss) for ss in atom_label]
    ln_sph = [len(ss) for ss in sph_label]
    atom_fmt = '%' + str(max(ln_atom)) + 's'
    sph_fmt = '%' + str(max(ln_sph)) + 's'
    hline = ''.join(['-' for i in range(0, 127)])
    hline_ = ''.join(['-- ' for i in range(0, 42)])
    nlarge = min(6, n_mo)
        
    for s in range(0, ns):
        if ns == 2:
            print('Orbital properties of spin-%s channel:' % ('alpha' if s==0 else 'beta'))
        elif ns == 1:
            print('Orbital properties:')
    
        #== Column headers ==#
        space1 = '    '
        print(hline)
        print((' %4s %14s %14s %' + str(10+of) + 's%s%s') % (
                'No.', 'Occupation', 'Energy', 'Irrep.', space1,
                'Six largest coefficients (value, center, spher.)'))
        print(hline)
        for i in range(0,n_mo):

            occ_s = 'N/A'
            if oocc is not None: occ_s = '%14.8f' % oocc_[s,i]
            erg_s = 'N/A'
            if oerg is not None: erg_s = '%14.8f' % oerg_[s,i]

            print((' %4d %14s %14s %' + str(10+of) + 's') % \
                  (i+1, occ_s, erg_s, osym_l[s][i]+' / '+str(osym[s][i])), end=space1)
    
            for j in range(0,nlarge):
                jj = int(idsort[s,j,i])
    
                if (j == int(nlarge/2)):
                    print('')
                    print(('%' + str(46+of) + 's') % '', end=space1)
                coeff_str = "{:.6f}".format(mo_c[s,jj,i])
                print('%s' % \
                      ('('+coeff_str+', '+atom_label[jj]+', '+sph_label[jj])+')', end='  ')
            print('')
            print(' '+hline_)    
##########################################################################


##########################################################################
def analyze_multipole(mol, orbs):

    #==== The multipole operator matrices in AO rep. ====#
    dpole_ao = mol.intor('int1e_r').reshape(3,mol.nao,mol.nao)
    qpole_ao = mol.intor('int1e_rr').reshape(3,3,mol.nao,mol.nao)

    #==== Column title ====#
    hline = ''.join(['-' for i in range(0, 7+11*9)])
    print(hline)
    print('%5s  %11s%11s%11s%11s%11s%11s%11s%11s%11s' %
          ('No.', 'x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz'))
    print(hline)

    #==== Multipole values ====#
    for i in range(0, mol.nao):
        dpole = np.einsum('j, xjk, k -> x', orbs[:,i], dpole_ao, orbs[:,i])
        qpole = np.einsum('j, xyjk, k -> xy', orbs[:,i], qpole_ao, orbs[:,i])
        print('%5d  ' % (i+1), end='')
        for j in range(0,3): print('%11.6f' % dpole[j], end='')
        for j in range(0,3): print('%11.6f' % np.diag(qpole,0)[j], end='')
        for j in range(0,2): print('%11.6f' % np.diag(qpole,1)[j], end='')
        for j in range(0,1): print('%11.6f' % np.diag(qpole,2)[j])

##########################################################################


##########################################################################
def analyze_population(mol, orbs, qtype='low'):

    #==== Obtain the range of AO id's for each atom ====#
    atom_ao_range = [None] * mol.natm
    start_found = False
    ia = 0
    for ia in range(0, mol.natm):
        for ib in range(0, mol.nao):
            ibm = min(ib+1, mol.nao-1)
            if mol.ao_labels(fmt=False)[ib][0] == ia:
                if not start_found:
                    ia_start = ib
                    start_found = True
                if mol.ao_labels(fmt=False)[ibm][0] == ia+1 or \
                   ib == mol.nao-1:
                    ia_last = ib
                    start_found = False
                    break
        atom_ao_range[ia] = (ia_start, ia_last)
    
    #==== Overlap matrix ====#
    ovl = mol.intor('int1e_ovlp')
    es, U = eigh(ovl)
    ovl_half = U @ (np.diag( np.sqrt(es) ) @ U.conj().T)

    #==== Calculate atomic populations ====#
    q = np.zeros((mol.nao, mol.natm))
    for i in range(0, mol.nao):
        #==== Density matrix ====#
        P = np.outer(orbs[:,i], orbs[:,i])

        #==== Population type ====#
        if qtype == 'low':
            T = np.einsum('ij, jk, ki -> i', ovl_half, P, ovl_half)
        elif qtype == 'mul':
            T = np.einsum('ij, ji -> i', P, ovl)
        else:
            raise ValueError(
                'analyze_population: The argument \'qtype\' has an undefined value. The ' + \
                'available options are \'low\' (Loewdin) or \'mul\' (Mulliken).')
        for ia in range(0, mol.natm):
            ib_1 = atom_ao_range[ia][0]
            ib_2 = atom_ao_range[ia][1]
    
            #==== Mulliken population ====#
            q[i,ia] = np.sum( T[ib_1:ib_2+1] )


    #==== Print atomic populations ====#
    maxcol = 10
    nblock = int(np.ceil(mol.natm/maxcol))
    j1 = 0
    j2 = min(maxcol, mol.natm) - 1
    for b in range(0, nblock):

        #== Column title ==#
        hline = ''.join(['-' for i in range(0, 7+11*min(maxcol, mol.natm))])
        print(hline)
        print('%5s  ' % 'No.', end='')
        for j in range(j1, j2+1):
            print('%11s' % (mol.atom_symbol(j) + str(j+1)), end='')
        print('')
        print(hline)

        #== Population values ==#
        for i in range(0, mol.nao):
            print('%5d  ' % (i+1), end='')
            for j in range(j1, j2+1):
                print('%11.6f' % q[i,j], end='')
            print('')
        print('')
        j1 += maxcol
        j2 += min(maxcol, mol.natm)

##########################################################################
