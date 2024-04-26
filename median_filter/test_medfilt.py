import pymedfiltmap 
import healpy as hp
import numpy as np  
from matplotlib import pyplot 

# Create a map with a single peak
nside = 64
npix = hp.nside2npix(nside)
m = np.random.normal(scale=0.1, size=npix)
radius = 5 * np.pi/180. 

pixels = hp.query_disc(nside, hp.ang2vec(np.pi/2,0), radius)
m[pixels] = 1

# Apply the median filter
m_med = pymedfiltmap.pymedfiltmap(m, radius*3)

# Plot the results
hp.mollview(m, title='Original map', sub=(1,2,1))
hp.mollview(m_med, title='Median filtered map', sub=(1,2,2))
pyplot.savefig('test_medfilt.png')
