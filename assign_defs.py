def_memory = 1E9
def_gs_noise = [1E-3]*4 + [1E-4]*4 + [0.0]*4
def_gs_dav_tols = [1E-3]*4 + [1E-6]*4 + [1E-8]*4
def_krylov_size = 20
def_exp_tol = 1.0E-6


def assign_defval(varname):
    try:
        val = eval(f'def_{varname}')
    except NameError:
        print(f'The parameter def_{varname} has not yet been defined inside assign_defs.py.')
    return val
