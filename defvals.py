
def_memory = 1E9
def_verbose_lvl = 4
def_dump_inputs = False
def_hf_orb_path = None
def_gs_steps = 50
def_gs_noise = [1E-3]*4 + [1E-4]*4 + [0.0]
def_gs_dav_tols = [1E-3]*4 + [1E-6]*4 + [1E-8]
def_gs_conv_tol = 1E-7
def_gs_cutoff = 1E-14
def_gs_bias = 1.0
def_inp_symmetry = 'c1'

def_ocoeff = None
def_ann_outmps_fname = 'ANN_KET'
def_ann_fit_margin = 150
def_ann_fit_noise = [1e-5]*4 + [1E-6]*4 + [0.0]
def_ann_fit_tol = 1E-14
def_ann_fit_steps = 30
def_ann_fit_cutoff = 1E-14
def_ann_fit_bias = 1.0
def_normalize_annout = True

def_te_method = 'tdvp'
def_exp_tol = 1.0E-6
def_te_cutoff = 0
def_te_normalize = False

def_krylov_size = 20
def_krylov_tol = 5.0E-6
def_n_sub_sweeps = 2
def_n_sub_sweeps_init = 4

def_te_sample = None
def_te_save_mps = True
def_te_save_1pdm = True
def_te_save_2pdm = False
