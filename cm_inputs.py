import sys
from os.path import exists
import numpy as np
import defvals


#######################################################
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
        inputs['inp_symmetry'] = inp_symmetry
    except NameError:
        inputs['inp_symmetry'] = defvals.def_inp_symmetry
    try:
        inputs['hf_orb_path'] = hf_orb_path
    except NameError:
        inputs['hf_orb_path'] = defvals.def_hf_orb_path
        
    #==== CAS parameters ====#
    inputs['nCore'] = nCore
    inputs['nCAS'] = nCAS
    inputs['nelCAS'] = nelCAS

    #==== Annihilation operation parameters ====#
    inputs['do_annihilate'] = do_annihilate
    if inputs['do_annihilate'] == True:
        inputs['D_gs'] = D_gs
        inputs['ann_sp'] = ann_sp
        inputs['ann_id'] = ann_id
        try:
            inputs['ann_outmps_dir'] = ann_outmps_dir
        except NameError:
            inputs['ann_outmps_dir'] = './' + inputs['prefix'] + '.annihilate_out'
        try:
            inputs['ann_outmps_fname'] = ann_outmps_fname
        except NameError:
            inputs['ann_outmps_fname'] = defvals.def_ann_outmps_fname
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
            inputs['gs_bias'] = gs_bias
        except NameError:
            inputs['gs_bias'] = defvals.def_gs_bias
        try:
            inputs['ocoeff'] = ocoeff
        except NameError:
            inputs['ocoeff'] = defvals.def_ocoeff
        try:
            inputs['ann_fit_margin'] = ann_fit_margin
        except NameError:
            inputs['ann_fit_margin'] = defvals.def_ann_fit_margin
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
            inputs['ann_fit_bias'] = ann_fit_bias
        except NameError:
            inputs['ann_fit_bias'] = defvals.def_ann_fit_bias
        try:
            inputs['normalize_annout'] = normalize_annout
        except NameError:
            inputs['normalize_annout'] = defvals.def_normalize_annout

    #==== Time evolution parameters ====#
    inputs['do_timeevo'] = do_timeevo
    if inputs['do_timeevo'] == True:
        inputs['te_max_D'] = te_max_D
        inputs['tmax'] = tmax
        inputs['dt'] = dt
        try:
            inputs['te_inmps_dir'] = te_inmps_dir
        except NameError:
            ann_dir = './' + inputs['prefix'] + '.annihilate_out'
            if exists(ann_dir):
                inputs['te_inmps_dir'] = ann_dir
            else:
                raise NameError('A mandatory input parameter \'te_inmps_dir\' has ' +
                                'not yet been specified.')
        try:
            inputs['te_inmps_fname'] = te_inmps_fname
        except NameError:
            inputs['te_inmps_fname'] = defvals.def_ann_outmps_fname        
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
