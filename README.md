# Nicer Mollweide projections


```python
src.mollweide
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
from mollweide.src import plot_mollweide
```
For examples of use see the example plots 
