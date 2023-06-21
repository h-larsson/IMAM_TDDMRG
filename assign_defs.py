


#######################################################
def assign_defval(varname):
    try:
        val = eval(f'def_{varname}')
    except NameError:
        print(f'The parameter def_{varname} has not yet been defined inside assign_defs.py.')
    return val
#######################################################


