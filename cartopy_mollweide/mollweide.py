import os 
import numpy as np
import healpy as hp
from matplotlib import pyplot

import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,LatitudeLocator)

import matplotlib.path as mpath
import matplotlib.colors as mcolors
from   matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from   matplotlib.ticker import AutoMinorLocator, FixedLocator, AsinhLocator, FuncFormatter

from astropy.wcs import WCS
from astropy.visualization import HistEqStretch, AsinhStretch, ImageNormalize,SqrtStretch, LogStretch, SinhStretch, LinearStretch
import colorcet as cc

from reproject import reproject_from_healpix

def plot_mollweide(m, 
                  cmap='cet_rainbow4', 
                  norm=None,
                  vmin = None,
                  vmax = None,
                  scale=1, 
                  cbar_ticks=None, 
                  grid=True,
                  filename=None, 
                  label=None,
                  bad_value_color='#D3D3D3',
                  grid_line_color='#5A5A5A',
                  coord=['G','G'], **kwargs):
    """
    Create a Mollweide projection of a HEALPix map

    Parameters
    ----------
    m : array-like
        HEALPix map to be plotted

    cmap : str
        Name of the colormap to be used
    
    norm : matplotlib.colors.Normalize
        Normalization to be used for the colormap (include astropy stretch)
    
    vmin : float
        Minimum value for the colormap

    vmax : float
        Maximum value for the colormap

    scale : float
        Scale the map by this factor

    cbar_ticks : array-like
        Ticks for the colorbar (default: AutoTickLocator)

    filename : str
        Name of the output file (default: None)

    label : str
        Label for the colorbar (default: None)

    bad_value_color : str
        Color for bad values (default: '#D3D3D3' - light gray)

    coord : array-like
        Coordinate system to be used for the projection (default: ['G','G'])



    """



    if isinstance(norm, type(None)):
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    npix = m.shape[-1]
    nside = hp.npix2nside(npix)

    # SCale the map
    m[m != hp.UNSEEN] *= scale

    # Get the projected Healpix map 
    fig = pyplot.figure(666)
    img_data = hp.cartview(m, title="", cbar=False, return_projected_map=True, hold=True, coord=coord, fig=666)
    pyplot.close(fig) 



    # Mollweide projection
    fig = pyplot.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=ccrs.Mollweide())

    # Adjust position of the main axes to make more space at the bottom
    axes_height = 0.9
    ax.set_position([0.025, 0.05, 0.85, axes_height])

    # Get the pixel boundaries
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lon = np.rad2deg(phi)
    lat = 90 - np.rad2deg(theta)

    cmap = pyplot.get_cmap(cmap)
    cmap.set_bad(bad_value_color)

    # Display the image
    img_extent = [-180, 180, -90, 90]
    img = ax.imshow(img_data, origin="lower", 
              extent=img_extent, 
              transform=ccrs.PlateCarree(), 
              interpolation="none", 
              cmap=cmap,
              norm=norm,
              aspect="auto")

    # Add gridlines with labels
    if grid:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                        draw_labels=False, 
                        linewidth=1, 
                        color=grid_line_color, 
                        alpha=0.5, linestyle='--')

    #cbar_ax = fig.add_axes([0.125, 0.125, 0.775, 0.02]) # horizontal colorbar
    # vertical colorbar on right hand side
    cbar_scale = 0.8
    cbar_height = axes_height * cbar_scale
    cbar_ax = fig.add_axes([0.88, 0.05 + axes_height*(1-cbar_scale)/2., 0.02 , cbar_height])

    cbar = fig.colorbar(img, ax=ax,cax=cbar_ax,  orientation="vertical")
    if not isinstance(label, type(None)):
        cbar.set_label(label,rotation=0,labelpad=10)
    # Update cbar ticks if cbar_ticks != None 
    if not isinstance(cbar_ticks, type(None)):
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    cbar.ax.tick_params(axis='y', direction='in')
    cbar.ax.tick_params(axis='y', which='minor', direction='in')

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pyplot.savefig(filename,dpi=300)

    return fig, ax, img, cbar


def plot_gnomview(m, 
                  rot,
                  width = 5,
                  cmap='cet_rainbow4', 
                  norm=None,
                  vmin = None,
                  vmax = None,
                  scale=1, 
                  cbar_ticks=None, 
                  grid=False,
                  filename=None, 
                  label=None,
                  bad_value_color='#D3D3D3',
                  grid_line_color='#5A5A5A',
                  coord=['G','G'], **kwargs):
    """
    Create a Gnomic projection of a HEALPix map

    Parameters
    ----------
    m : array-like
        HEALPix map to be plotted

    rot : array-like
        Central coordinate of the projection (lon, lat) in degrees

    width : float
        Width of the projected map in degrees

    cmap : str
        Name of the colormap to be used
    
    norm : matplotlib.colors.Normalize
        Normalization to be used for the colormap (include astropy stretch)
    
    vmin : float
        Minimum value for the colormap

    vmax : float
        Maximum value for the colormap

    scale : float
        Scale the map by this factor

    cbar_ticks : array-like
        Ticks for the colorbar (default: AutoTickLocator)

    filename : str
        Name of the output file (default: None)

    label : str
        Label for the colorbar (default: None)

    bad_value_color : str
        Color for bad values (default: '#D3D3D3' - light gray)

    coord : array-like
        Coordinate system to be used for the projection (default: ['G','G'])



    """



    if isinstance(norm, type(None)):
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())

    npix = m.shape[-1]
    nside = hp.npix2nside(npix)

    pixel_size = np.degrees(hp.nside2resol(nside))
    npix_projected = int(width / pixel_size)
    shape_out = (npix_projected, npix_projected)

    # SCale the map
    m[m != hp.UNSEEN] *= scale

    # Get the projected Healpix map 
    if coord[0] == 'G':
        coord_type = 'galactic'
        coord_ctype = ['GLON-', 'GLAT-']
        coord_labels = ['Galactic Longitude', 'Galactic Latitude']
    else:
        coord_type = 'icrs'
        coord_ctype = ['RA---', 'DEC--']
        coord_labels = ['Right Ascension', 'Declination']
    plot_wcs = WCS(naxis=2)
    plot_wcs.wcs.crpix = [npix_projected/2, npix_projected/2]
    plot_wcs.wcs.cdelt = [-pixel_size, pixel_size]
    plot_wcs.wcs.crval = rot
    plot_wcs.wcs.ctype = [f'{coord_ctype[0]}TAN', f'{coord_ctype[1]}TAN']

    img_data, footprint = reproject_from_healpix((m, coord_type), 
                                                plot_wcs.to_header(),
                                                shape_out=shape_out, 
                                                nested=False, 
                                                order='nearest-neighbor')
    img_data[img_data == hp.UNSEEN] = np.nan

    # Mollweide projection
    fig = pyplot.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection=plot_wcs)

    cmap = pyplot.get_cmap(cmap)
    cmap.set_bad(bad_value_color)

    # Display the image
    img = ax.imshow(img_data, origin="lower", 
              interpolation="none", 
              cmap=cmap,
              norm=norm,
              aspect="auto")
    
    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    if grid: 
        ax.coords.grid(color=grid_line_color, alpha=0.5, linestyle='--')

    # vertical colorbar on right hand side
    axes_height = ax.get_position().height
    axes_bottom = ax.get_position().y0
    cbar_scale = 0.8
    cbar_height = axes_height * cbar_scale
    cbar_ax = fig.add_axes([0.91, axes_bottom + axes_height*(1-cbar_scale)/2., 0.02 , cbar_height])

    cbar = fig.colorbar(img, ax=ax,cax=cbar_ax,  orientation="vertical")
    if not isinstance(label, type(None)):
        cbar.set_label(label,rotation=0,labelpad=10)
    # Update cbar ticks if cbar_ticks != None 
    if not isinstance(cbar_ticks, type(None)):
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    cbar.ax.tick_params(axis='y', direction='in')
    cbar.ax.tick_params(axis='y', which='minor', direction='in')

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pyplot.savefig(filename,dpi=300)

    return fig, ax, img, cbar
