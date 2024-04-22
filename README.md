# Nicer Moll/Gnom/Arc plots


Example usage: 

```python
from healpix_tools import Mollview, Gnomview, Arcview
from plotting_tools import MOLLVIEW_FIGSIZE
from matplotlib import pyplot
import healpy as hp 

FILENAME = 'map.fits'
m = hp.read_map(FILENAME) 
moll = Mollview()
figure = pyplot.figure(figsize=MOLLVIEW_FIGSIZE) 
moll(m, vmin=0, vmax=1)
moll.add_grid() 
moll.add_colorbar()
moll.add_contour(contour_map, levels=[0,1,2])
fig.savefig('test.png')
pyplot.close(fig)
```


# Nicer Mollweide projections (With cartopy -- don't use)


```python
mollweide.mollweide
```

contains two functions for plotting nice "mollweide" projections and "gnomonic" projections with the functions

```python]
mollweide.plot_mollweide 
mollweide.plot_gnomview 
```

# Install 

Make sure your environment has the required packages (reproject and cartopy) by running:

```bash
pip install -r requirements.txt 
```

move the mollweide.py script to your working directory or run the setup.py script as:

```bash
python setup.py install 
```

and then you can import the functions using

```python
from mollweide.mollweide import plot_mollweide
```
For examples of use see the example plots 
