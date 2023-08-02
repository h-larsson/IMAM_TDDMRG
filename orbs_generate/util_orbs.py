import numpy as np


##########################################################################
def sort_orbs(orbs, occs, ergs, sr='erg', s='de'):
    '''
    Sort occs, orbs, and ergs based on the elements of occs or ergs.
    '''

    assert isinstance(occs, np.ndarray) or occs is None
    assert isinstance(ergs, np.ndarray) or ergs is None

    
    assert sr == 'erg' or sr == 'occ', \
        'sort_orbs: The value of the argument \'sr\' can only either ' + \
        'be \'erg\' or \'occ\'. Its current value: sr = ' + sr + '.'
    assert s == 'de' or s == 'as', \
        'sort_orbs: The value of the argument \'s\' can only either ' + \
        'be \'de\' or \'as\'. Its current value: s = ' + s + '.'

    if sr == 'erg' and ergs is None:
        raise ValueError('sort_orbs: If sr = \'erg\', then ergs must ' + \
                         'not be None.')
    if sr == 'occ' and occs is None:
        raise ValueError('sort_orbs: If sr = \'occ\', then occs must ' + \
                         'not be None.')

    if s == 'de':       # Descending
        if sr == 'erg': isort = np.argsort(-ergs)
        if sr == 'occ': isort = np.argsort(-occs)
    elif s == 'as':     # Ascending
        if sr == 'erg': isort = np.argsort(ergs)
        if sr == 'occ': isort = np.argsort(occs)
    
    if occs is not None: occs = occs[isort]
    if ergs is not None: ergs = ergs[isort]
    orbs = orbs[:,isort]

    return orbs, occs, ergs
##########################################################################


