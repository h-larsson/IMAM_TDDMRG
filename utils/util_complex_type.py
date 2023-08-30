import inspect




isset = False
CX_TYPE = None

def check_type():
    assert CX_TYPE == 'full' or CX_TYPE == 'hybrid'


def init(cx):
    global isset, CX_TYPE
    assert not isset
    CX_TYPE = cx
    check_type()
    isset = True
    

def get_complex_type():
    caller_file = inspect.stack()[1].filename
    if inspect.stack()[1].function == '<module>':
        assert isset, \
            'The util_complex_type module called from inside ' + caller_file + \
            ' has not been initiated yet. Call util_complex_type.init to initiate.'
    else:
        caller_func = inspect.stack()[1].function
        assert isset, \
        'The util_complex_type module called by the function ' + caller_func + \
        ' inside ' + caller_file + ' has not been initiated yet. Call ' + \
        'util_complex_type.init to initiate.'
    
    check_type()
    return CX_TYPE
