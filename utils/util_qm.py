import numpy as np



#################################################
def make_full_dm(ncore, dm):
    assert len(dm.shape) == 3
    assert dm.shape[0] == 2
    assert dm.shape[1] == dm.shape[2]
    
    complex_dm = (type(dm[0,0,0]) == np.complex128)
    dtype = np.complex128 if complex_dm else np.float64
    nact = dm.shape[2]
    nocc = ncore + nact
    
    dm_ = np.zeros((2, nocc, nocc), dtype=dtype)
    for i in range(0,2):
        if complex_dm:
            dm_[i, 0:ncore, 0:ncore] = np.diag( [complex(1.0, 0.0)]*ncore )
        else:
            dm_[i, 0:ncore, 0:ncore] = np.diag( [1.0]*ncore )
        dm_[i, ncore:nocc, ncore:nocc] = dm[i, :, :]

    return dm_
#################################################
