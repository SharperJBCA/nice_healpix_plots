rm pymedfiltmap.cpython-311-x86_64-linux-gnu.so 
HEALPIX=/path/to/Healpix_3.81

f2py pymedfiltmap.f90 -m pymedfiltmap -h pymedfiltmap.pyf 
f2py -c  pymedfiltmap.pyf pymedfiltmap.f90 -L$HEALPIX/lib/ -I$HEALPIX/include/ -lhealpix -lgomp -lcfitsio --f90flags='-fopenmp'
python test_medfilt.py