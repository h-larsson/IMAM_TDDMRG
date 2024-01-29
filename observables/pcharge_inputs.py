from IMAM_TDDMRG.observables import observables_defvals as defvals


#######################################################
def get_inputs(inp_file):
    
    exec(open(inp_file).read(), globals())

    inputs = {}

    #==== General parameters ====#
    inputs['inp_coordinates'] = inp_coordinates
    inputs['inp_basis'] = inp_basis
    inputs['orb_path'] = orb_path
    inputs['nCore'] = nCore
    inputs['nCAS'] = nCAS
    inputs['nelec_t0'] = nelec_t0

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
        inputs['sample_dir'] = sample_dir
    except NameError:
        inputs['sample_dir'] = 'DEFINE_LATER'
    try:
        inputs['save_txt'] = save_txt
    except NameError:
        inputs['save_txt'] = defvals.def_save_txt
    try:
        inputs['save_npy'] = save_npy
    except NameError:
        inputs['save_npy'] = defvals.def_save_npy
        
    return inputs
#######################################################
