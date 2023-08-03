import sys
from os.path import exists
import numpy as np
import defvals


#######################################################
def get_inputs(inp_file):
    '''
    This function returns a dictionary containing the input parameters and their values
    as key-value pairs.
    
    To define a new input parameters, first decide whether this new input parameter is 
    mandatory or optional parameter. If it is a mandatory input, then simply add the 
    following line after the 'inputs = {}' below
       inputs['new_input'] = new_input
    where new_input is to be replaced with the intended name of the new input. On the
    other hand, if the new input is optional, do the following steps.
       1) If the default value of this optional input does not depend on any other 
          parameters, then add the following line anywhere after the 'inputs = {}' line 
          below
             try:
                 inputs['new_input'] = new_input
             except NameError:
                 inputs['new_input'] = defvals.def_new_input
          Then, add this line inside defvals.py
             def_new_input = <value>
          where def_new_input is to be replaced with the intended name of the variable that
          holds the default value for this optional input, and <value> is the default value.
       2) If the default value depends on some parameters and can be determined here (e.g. 
          the input prefix below, whose default value depends on the input file name), then 
          replace the inputs['new_input'] = defvals.def_new_input line in point 1) 
          above with the appropriate line(s) that involve the dependency parameters.
          The addition of the corresponding default parameter inside defvals.py is 
          unnecessary in this case since the default value has been calculated here.
       3) If the default value depends on some parameters and can only be determined later
          in the program, then replace the right-hand side of the 
          inputs['new_input'] = defvals.def_new_input line in point 1) above with the string
          'DEFINE_LATER'. And later in the code when this input is about to be used for the 
          first time, do the assignment in the following way:
             if inputs['new_input'] == 'DEFINE_LATER':
                 <the appropriate lines to determine the value of inputs['new_input']>
    
    The newly defined input parameter can then be referred to by invoking inputs['new_input'] 
    inside cm_dmrg assuming that the output of this function is stored in a variable called 
    inputs. 
    
    It is strongly encouraged that any newly defined inputs are added such that all 
    mandatory inputs are defined before all optional inputs in their approppriate section 
    (marked by a comment line such as #=== abcde ===# below).
    '''
    
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
        inputs['memory'] = memory
    except NameError:
        inputs['memory'] = defvals.def_memory
    try:
        inputs['prefix'] = prefix
    except NameError:
        if inp_file[len(inp_file)-3:len(inp_file)] == '.py':
            inputs['prefix'] = inp_file[0:len(inp_file)-3]
        else:
            inputs['prefix'] = inp_file
    try:
        inputs['verbose_lvl'] = verbose_lvl
    except NameError:
        inputs['verbose_lvl'] = defvals.def_verbose_lvl
    try:
        inputs['inp_ecp'] = inp_ecp
    except NameError:
        inputs['inp_ecp'] = defvals.def_inp_ecp
    try:
        inputs['inp_symmetry'] = inp_symmetry
    except NameError:
        inputs['inp_symmetry'] = defvals.def_inp_symmetry
    try:
        inputs['orb_path'] = orb_path
    except NameError:
        inputs['orb_path'] = defvals.def_orb_path
    try:
        inputs['orb_order'] = orb_order
    except NameError:
        inputs['orb_order'] = 'DEFINE_LATER'
        
    #==== CAS parameters ====#
    inputs['nCore'] = nCore
    inputs['nCAS'] = nCAS
    inputs['nelCAS'] = nelCAS
    inputs['twos'] = twos

    #==== Ground state parameters ====#
    inputs['do_groundstate'] = do_groundstate
    if inputs['do_groundstate'] == True:
        inputs['D_gs'] = D_gs
        try:
            inputs['gs_noise'] = gs_noise
        except NameError:
            inputs['gs_noise'] = defvals.def_gs_noise
        try:
            inputs['gs_dav_tols'] = gs_dav_tols
        except NameError:
            inputs['gs_dav_tols'] = defvals.def_gs_dav_tols
        try:
            inputs['gs_steps'] = gs_steps       # Maximum number of iteration steps
        except NameError:
            inputs['gs_steps'] = defvals.def_gs_steps
        try:
            inputs['gs_conv_tol'] = gs_conv_tol
        except NameError:
            inputs['gs_conv_tol'] = defvals.def_gs_conv_tol
        try:
            inputs['gs_cutoff'] = gs_cutoff
        except NameError:
            inputs['gs_cutoff'] = defvals.def_gs_cutoff
        try:
            inputs['gs_occs'] = gs_occs
        except NameError:
            inputs['gs_occs'] = defvals.def_gs_occs
        try:
            inputs['gs_bias'] = gs_bias
        except NameError:
            inputs['gs_bias'] = defvals.def_gs_bias
        try:
            inputs['gs_outmps_dir'] = gs_outmps_dir
        except NameError:
            inputs['gs_outmps_dir'] = 'DEFINE_LATER'
        try:
            inputs['gs_outmps_fname'] = gs_outmps_fname
        except NameError:
            inputs['gs_outmps_fname'] = defvals.def_gs_outmps_fname
        try:
            inputs['save_gs_1pdm'] = save_gs_1pdm
        except NameError:
            inputs['save_gs_1pdm'] = defvals.def_save_gs_1pdm
    
    #==== Annihilation operation parameters ====#
    inputs['do_annihilate'] = do_annihilate
    if inputs['do_annihilate'] == True:
        inputs['ann_sp'] = ann_sp
        inputs['ann_orb'] = ann_orb
        inputs['D_ann_fit'] = D_ann_fit
        try:
            inputs['ann_inmps_dir'] = ann_inmps_dir
        except NameError:
            inputs['ann_inmps_dir'] = 'DEFINE_LATER'
        try:
            inputs['ann_inmps_fname'] = ann_inmps_fname
        except NameError:
            inputs['ann_inmps_fname'] = defvals.def_ann_inmps_fname
        try:
            inputs['ann_outmps_dir'] = ann_outmps_dir
        except NameError:
            inputs['ann_outmps_dir'] = 'DEFINE_LATER'
        try:
            inputs['ann_outmps_fname'] = ann_outmps_fname
        except NameError:
            inputs['ann_outmps_fname'] = defvals.def_ann_outmps_fname
        try:
            inputs['ann_orb_thr'] = ann_orb_thr
        except NameError:
            inputs['ann_orb_thr'] = defvals.def_ann_orb_thr
        #OLD try:
        #OLD     inputs['ann_fit_margin'] = ann_fit_margin
        #OLD except NameError:
        #OLD     inputs['ann_fit_margin'] = defvals.def_ann_fit_margin
        try:
            inputs['ann_fit_noise'] = ann_fit_noise
        except NameError:
            inputs['ann_fit_noise'] = defvals.def_ann_fit_noise
        try:
            inputs['ann_fit_tol'] = ann_fit_tol
        except NameError:
            inputs['ann_fit_tol'] = defvals.def_ann_fit_tol
        try:
            inputs['ann_fit_steps'] = ann_fit_steps
        except NameError:
            inputs['ann_fit_steps'] = defvals.def_ann_fit_steps
        try:
            inputs['ann_fit_cutoff'] = ann_fit_cutoff
        except NameError:
            inputs['ann_fit_cutoff'] = defvals.def_ann_fit_cutoff
        try:
            inputs['ann_fit_occs'] = ann_fit_occs
        except NameError:
            inputs['ann_fit_occs'] = defvals.def_ann_fit_occs
        try:
            inputs['ann_fit_bias'] = ann_fit_bias
        except NameError:
            inputs['ann_fit_bias'] = defvals.def_ann_fit_bias
        try:
            inputs['normalize_annout'] = normalize_annout
        except NameError:
            inputs['normalize_annout'] = defvals.def_normalize_annout
        try:
            inputs['save_ann_1pdm'] = save_ann_1pdm
        except NameError:
            inputs['save_ann_1pdm'] = defvals.def_save_ann_1pdm

    #==== Time evolution parameters ====#
    inputs['do_timeevo'] = do_timeevo
    if inputs['do_timeevo'] == True:
        inputs['te_max_D'] = te_max_D
        inputs['tmax'] = tmax
        inputs['dt'] = dt
        try:
            inputs['te_inmps_dir'] = te_inmps_dir
        except NameError:
            inputs['te_inmps_dir'] = 'DEFINE_LATER'
        try:
            inputs['te_inmps_fname'] = te_inmps_fname
        except NameError:
            inputs['te_inmps_fname'] = defvals.def_te_inmps_fname        
        try:
            inputs['te_method'] = te_method
        except NameError:
            inputs['te_method'] = defvals.def_te_method
        try:
            inputs['exp_tol'] = exp_tol
        except NameError:
            inputs['exp_tol'] = defvals.def_exp_tol
        try:
            inputs['te_cutoff'] = te_cutoff
        except NameError:
            inputs['te_cutoff'] = defvals.def_te_cutoff
        try:
            inputs['krylov_size'] = krylov_size
        except NameError:
            inputs['krylov_size'] = defvals.def_krylov_size
        try:
            inputs['krylov_tol'] = krylov_tol
        except NameError:
            inputs['krylov_tol'] = defvals.def_krylov_tol
        try:
            inputs['n_sub_sweeps'] = n_sub_sweeps
        except NameError:
            inputs['n_sub_sweeps'] = defvals.def_n_sub_sweeps
        try:
            inputs['n_sub_sweeps_init'] = n_sub_sweeps_init
        except NameError:
            inputs['n_sub_sweeps_init'] = defvals.def_n_sub_sweeps_init
        try:
            inputs['te_normalize'] = te_normalize
        except NameError:
            inputs['te_normalize'] = defvals.def_te_normalize
        try:
            inputs['te_sample'] = te_sample
        except NameError:
            inputs['te_sample'] = defvals.def_te_sample
        try:
            inputs['te_save_mps'] = te_save_mps
        except NameError:
            inputs['te_save_mps'] = defvals.def_te_save_mps
        try:
            inputs['te_save_1pdm'] = te_save_1pdm
        except NameError:
            inputs['te_save_1pdm'] = defvals.def_te_save_1pdm
        try:
            inputs['te_save_2pdm'] = te_save_2pdm
        except NameError:
            inputs['te_save_2pdm'] = defvals.def_te_save_2pdm

    return inputs
#######################################################
