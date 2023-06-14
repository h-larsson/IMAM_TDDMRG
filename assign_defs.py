def_memory = 1E9
def_exp_tol = 1.0E-6


def assign_defval(varname):
    try:
        val = eval(f'def_{varname}')
    except NameError:
        print(f'The parameter def_{varname} has not yet been defined inside assign_defs.py.')
    return val
