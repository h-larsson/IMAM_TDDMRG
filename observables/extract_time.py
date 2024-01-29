import glob
import numpy as np


def get(sample_dir):

    pdm_dir = glob.glob(sample_dir + '/tevo-*')
    n_t = len(pdm_dir)

    tt = np.zeros(n_t)
    for i in range(0, n_t):
        with open(pdm_dir[i] + '/TIME_INFO', 'r') as t_info:
            lines = t_info.read()
            line0 = lines.split('\n')
            for l in line0:
                if 'Actual sampling time' in l:
                    tt[i] = float( l.split()[5] )

    return tt

