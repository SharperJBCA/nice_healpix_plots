# SOME CONSTANTS FOR PLOTTING
#### 
# These have been setup so things look good for MNRAS
#
# If you want the MNRAS font download it from: https://www.fontsquirrel.com/fonts/nimbus-roman-no9-l
# Install on your system and then set it as the default font using:
# pyplot.rcParams.update({
#    'font.family': 'Nimbus Roman No9 L'})
# Or update your matplotlibrc file to include the line:
# font.family : Nimbus Roman No9 L


MOLLVIEW_FIGSIZE = (127./18.,127./18./2.) # Based on textwidth in MNRAS being 508pt and aspect ratio of 2:1
MOLLVIEW_FIGSIZE_TRIPLE = (127./18./3.,127./18./2./3.) # Based on textwidth in MNRAS being 508pt and aspect ratio of 2:1
MOLLVIEW_FIGSIZE_DOUBLE = (127./18./2.,127./18./2./2.) # Based on textwidth in MNRAS being 508pt and aspect ratio of 2:1

COLUMN_FIGSIZE = (10./3., 3.) # Based on column width in MNRAS being 240pt 
THREE_COLUMN_FIGSIZE = (10./3.* 2./3., 3. * 2./3.) # Based on column width in MNRAS being 240pt
TEXT_FIGSIZE_60 = (127./18.*1.25,127./18.*0.6*1.25) # Based on column width in MNRAS being 240pt 

DATA_TITLES = {'Planck30_PR4':'Planck 28.1 GHz',
               'Planck30_COSMO':'Planck 28.1 GHz',
               'Planck30_PR3':'Planck 28.1 GHz',
               'WMAP_K_9yr':'WMAP 22.5 GHz',
               'WMAP_K_COSMO':'WMAP 22.5 GHz',
               'CBASS':'C-BASS 4.76 GHz',
               'SPASS':'S-PASS 2.3 GHz'}


FONT_SIZE_SMALL = 7
FONT_SIZE_NORMAL = 8  # Default for MNRAS is 9pt - after checking, it looks like 8pt might be better?
FONT_SIZE_LARGE = 9
