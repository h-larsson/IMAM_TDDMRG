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
    try:
        inputs['loc_orb'] = loc_orb
    except NameError:
        inputs['loc_orb'] = defvals.def_loc_orb
    if inputs['loc_orb']:
        try:
            inputs['loc_type'] = loc_type
        except NameError:
            inputs['loc_type'] = defvals.def_loc_type
        try:
            inputs['loc_irrep'] = loc_irrep
        except NameError:
            inputs['loc_irrep'] = defvals.def_loc_irrep
    
        
    #==== CAS parameters ====#
    inputs['source'] = source
    if inputs['source'] == 'casscf':
        inputs['nCAS'] = nCAS
        inputs['nelCAS'] = nelCAS

        try:
            inputs['frozen'] = frozen
        except NameError:
            inputs['frozen'] = defvals.def_frozen
        try:
            inputs['init_orbs'] = init_orbs
        except NameError:
            inputs['init_orbs'] = defvals.def_init_orbs
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
        if inputs['loc_orb']:
            try:
                inputs['loc_thr'] = loc_thr
            except NameError:
                inputs['loc_thr'] = defvals.def_loc_thr
        try:
            inputs['fcisolver'] = fcisolver
        except NameError:
            inputs['fcisolver'] = defvals.def_fcisolver
        if inputs['fcisolver'] is not None:
            inputs['max_bond_dim'] = max_bond_dim
