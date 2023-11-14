import orbs_generate.defvals as defvals


def get_inputs(inp_file):

    exec(open(inp_file).read(), globals())
    inputs = {}

    #==== General parameters ====#
    inputs['inp_coordinates'] = inp_coordinates
    inputs['inp_basis'] = inp_basis

    try:
        inputs['dump_inputs'] = dump_inputs
    except NameError:
        inputs['dump_inputs'] = defvals.def_dump_inputs
    try:
        inputs['prefix'] = prefix
    except NameError:
        if inp_file[len(inp_file)-3:len(inp_file)] == '.py':
            inputs['prefix'] = inp_file[0:len(inp_file)-3]
        else:
            inputs['prefix'] = inp_file
    try:
        inputs['inp_ecp'] = inp_ecp
    except NameError:
        inputs['inp_ecp'] = defvals.def_inp_ecp
    try:
        inputs['inp_symmetry'] = inp_symmetry
    except NameError:
        inputs['inp_symmetry'] = defvals.def_inp_symmetry
    try:
        inputs['save_dir'] = save_dir
    except NameError:
        inputs['save_dir'] = defvals.def_save_dir

    try:
        inputs['sz'] = sz
    except NameError:
        inputs['sz'] = defvals.def_sz
    try:
        inputs['natorb'] = natorb
    except NameError:
        inputs['natorb'] = defvals.def_natorb

        
    #==== CAS parameters ====#
    try:
        inputs['source'] = source
    except NameError:
        inputs['source'] = defvals.def_source
        
    if inputs['source'] == 'rhf':
        # All inputs used by RHF are also used by CASSCF.
        pass
    elif inputs['source'] == 'casscf':
        inputs['nCAS'] = nCAS
        inputs['nelCAS'] = nelCAS
        inputs['init_orbs'] = init_orbs

        try:
            inputs['frozen'] = frozen
        except NameError:
            inputs['frozen'] = defvals.def_frozen
        try:            
            inputs['ss'] = ss
        except NameError:
            inputs['ss'] = defvals.def_ss
        try:            
            inputs['ss_shift'] = ss_shift
        except NameError:
            inputs['ss_shift'] = defvals.def_ss_shift
        try:
            inputs['wfnsym'] = wfnsym
        except NameError:
            inputs['wfnsym'] = defvals.def_wfnsym
        try:
            inputs['fcisolver'] = fcisolver
        except NameError:
            inputs['fcisolver'] = defvals.def_fcisolver
        try:
            inputs['init_basis'] = init_basis
        except NameError:
            inputs['init_basis'] = defvals.def_init_basis
        try:
            inputs['sort_out'] = sort_out
        except NameError:
            inputs['sort_out'] = defvals.def_sort_out

        #==== DMRGSCF ====#
        try:
            inputs['max_bond_dim'] = max_bond_dim
        except NameError:
            inputs['max_bond_dim'] = defvals.def_max_bond_dim
        try:
            inputs['sweep_tol'] = sweep_tol
        except NameError:
            inputs['sweep_tol'] = defvals.def_sweep_tol

    elif inputs['source'] == 'dft':
        inputs['xc'] = xc
                

    try:
        inputs['localize'] = localize
    except NameError:
        inputs['localize'] = defvals.def_localize
        
    if inputs['localize']:
        inputs['loc_subs'] = loc_subs

        try:
            inputs['orbs_for_loc'] = orbs_for_loc
        except NameError:
            inputs['orbs_for_loc'] = defvals.def_orbs_for_loc
        try:
            inputs['rdm_for_loc'] = rdm_for_loc
        except NameError:
            inputs['rdm_for_loc'] = defvals.def_rdm_for_loc
        try:
            inputs['loc_type'] = loc_type
        except NameError:
            inputs['loc_type'] = ['PM'] * len(inputs['loc_subs'])
        try:
            inputs['loc_irrep'] = loc_irrep
        except NameError:
            inputs['loc_irrep'] = [True] * len(inputs['loc_subs'])
        
        try:
            inputs['update_loc_occs'] = update_loc_occs
        except NameError:
            inputs['update_loc_occs'] = defvals.def_update_loc_occs
        try:
            inputs['update_loc_rdm'] = update_loc_rdm
        except NameError:
            inputs['update_loc_rdm'] = defvals.def_update_loc_rdm
        
    return inputs
