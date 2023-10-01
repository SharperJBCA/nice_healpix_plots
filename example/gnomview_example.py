# update the python path to include the src directory
import os 
import sys

import numpy as np
import healpy as hp 
from matplotlib import pyplot
try:
    from mollweide import mollweide as mw
except ImportError:
    sys.path.append(os.path.abspath('../'))
    from mollweide import mollweide as mw

# Set serif fonts
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 12

if __name__ == "__main__":
    
    nside = 64
    test_map = np.random.normal(size=hp.nside2npix(nside)) 
    test_mask_pixels = hp.query_strip(nside, np.radians(30), np.radians(60))
    test_map[test_mask_pixels] = hp.UNSEEN 

    # plot the map
    mw.plot_gnomview(test_map, 
                    [0,30],
                    grid=True,
                    label='K',
                    width = 5)
    pyplot.show()
