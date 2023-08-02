import numpy as np


##########################################################################
def print_matrix(m, maxcol=10):

    nrow = m.shape[0]
    ncol = m.shape[1]
    
    nblock = int(np.ceil(ncol/maxcol))
    j1 = 0
    j2 = min(maxcol, ncol) - 1
    for b in range(0, nblock):

        #== Column title ==#
        print('%5s   ' % '', end='')
        for j in range(j1, j2+1):
            print('%11d' % (j+1), end='')
        print('')
        if b < nblock-1:
            hline = ''.join(['-' for i in range(0, 11*maxcol)])
        elif b == nblock-1:
            hline = ''.join(['-' for i in range(0, 11*(j2-j1+1))])
        print('%5s   ' % '', end='')
        print(hline)

        #== Population values ==#
        for i in range(0, nrow):
            print('%5d  |' % (i+1), end='')
            for j in range(j1, j2+1):
                print('%11.6f' % m[i,j], end='')
            print('')
        print('')
        j1 += maxcol
        j2 = min(j1+maxcol, ncol) - 1

##########################################################################
