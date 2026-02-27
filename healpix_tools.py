#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:55:51 2023

@author: sharper
"""
# import as necessary
import numpy as np
import healpy as hp
from matplotlib import pyplot
from types import GenericAlias 
from astropy.wcs import WCS 
from dataclasses import dataclass, field 

from matplotlib.figure import Figure 
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy.visualization import simple_norm, HistEqStretch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.coordinates import SkyCoord
import astropy.units as u
from plotting_tools import FONT_SIZE_NORMAL

from reproject import reproject_from_healpix
from astropy.visualization import (ManualInterval,MinMaxInterval, AsinhStretch, SqrtStretch, HistEqStretch, ImageNormalize, LinearStretch)

#from cmcrameri import cm
from matplotlib import cm

def todegrees(theta,phi):
    return np.degrees(phi), np.degrees(np.pi/2-theta)

def tothetaphi(x,y):
    return np.pi/2-y*np.pi/180., x*np.pi/180.

HealpixMap = GenericAlias(np.ndarray, (float,))    


class _HealpixViewBase:
    """Shared helpers for HEALPix-backed WCS views."""

    def _prepare_map_for_reproject(self, m: HealpixMap):
        """Return a copy of ``m`` with invalid/unseen values masked for reprojection."""
        m_clean = np.array(m, copy=True)
        invalid = (~np.isfinite(m_clean)) | (m_clean == hp.UNSEEN) | (np.abs(m_clean) > 1e10) | (m_clean < -1e20)
        m_clean[invalid] = hp.UNSEEN
        return m_clean

    def _reproject_healpix(self, m: HealpixMap, wcs: WCS, shape_out, interpolation=None):
        """Reproject a HEALPix map into a target WCS."""
        interp = interpolation or self.interpolation
        map_clean = self._prepare_map_for_reproject(m)
        array, footprint = reproject_from_healpix(
            (map_clean, 'galactic'),
            wcs,
            shape_out=shape_out,
            nested=False,
            order=interp,
        )
        array[(array == hp.UNSEEN) | (~np.isfinite(array)) | (np.abs(array) > 1e10)] = np.nan
        return array, footprint

    def _resolve_vmin_vmax(self, array, vmin, vmax):
        """Resolve numeric and percentile-style vmin/vmax values."""
        def _resolve(value):
            if isinstance(value, str):
                text = value.strip()
                if text.endswith('%'):
                    text = text[:-1]
                if text.lower().startswith('p'):
                    text = text[1:]
                return np.nanpercentile(array, float(text))
            return value

        return _resolve(vmin), _resolve(vmax)

    def _build_norm(self, array, vmin, vmax, asinh=False, norm_mode=None):
        """Build a matplotlib/astropy normalization object."""
        if asinh:
            return ImageNormalize(array, interval=ManualInterval(vmin=vmin, vmax=vmax), stretch=AsinhStretch(a=0.1))
        if norm_mode in (None, 'linear'):
            return ImageNormalize(array, interval=ManualInterval(vmin=vmin, vmax=vmax), stretch=LinearStretch())
        if norm_mode == 'hist':
            finite = array[np.isfinite(array)]
            return ImageNormalize(array, interval=MinMaxInterval(), stretch=HistEqStretch(finite if finite.size else np.array([0.0])))
        return simple_norm(array, norm_mode, min_cut=vmin, max_cut=vmax)

    def _imshow(self, array, cmap=None, interpolation='nearest', vmin=None, vmax=None, asinh=False, norm_mode=None):
        """Consistent imshow wrapper across projections."""
        vmin, vmax = self._resolve_vmin_vmax(array, vmin, vmax)
        norm = self._build_norm(array, vmin=vmin, vmax=vmax, asinh=asinh, norm_mode=norm_mode)
        self.img = self.axes.imshow(array, cmap=cmap, norm=norm, interpolation=interpolation, origin='lower')
        return self.img

@dataclass 
class Mollview(_HealpixViewBase):
    
    map : HealpixMap = field(default_factory=lambda : np.zeros(1)) 
    wcs : WCS = field(default_factory=lambda : WCS(naxis=2)) 
    Nx : int = 0
    Ny : int = 0
    interpolation : str = 'nearest-neighbor' 
    
    # Matplotlib info 
    axes : Axes = field(default_factory = lambda : None )
    figure : Figure = field(default_factory = lambda : None )
    
    def __post_init__(self): 
        
        
        # build wcs first
    
        cdelt = 1./6
        self.Nx,self.Ny = int(360//cdelt*0.895), int(180//cdelt*0.895)
        self.wcs.wcs.crpix=[self.Nx//2,self.Ny//2+2]
        self.wcs.wcs.cdelt=[-cdelt, cdelt]
        self.wcs.wcs.crval=[0,0]
        self.wcs.wcs.ctype=['GLON-MOL','GLAT-MOL']
        
    def __call__(self, m : HealpixMap,
                 axes = None, 
                 figure = None, 
                 norm : str =None, 
                 asinh : bool = False,
                 vmin : float =None, vmax : float =None, cmap=cm.viridis): 
        """
        

        Parameters
        ----------
        m : HealpixMap
            DESCRIPTION.
        norm : str, optional
            DESCRIPTION. The default is None.
        vmin : float, optional
            DESCRIPTION. The default is None.
        vmax : float, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is cm.batlow_r.

        Returns
        -------
        None.

        """
        
        self.load_data(m)
        return self.plot_data(axes=axes, figure=figure, norm=norm, asinh=asinh, vmin=vmin, vmax=vmax, cmap=cmap)

    def load_data(self, healpix_map: HealpixMap):
        """Reproject and cache a HEALPix map for plotting."""
        self.array, self.footprint = self._reproject_healpix(healpix_map, self.wcs, [self.Ny, self.Nx], self.interpolation)
        return self.array

    def plot_data(self, axes=None, figure=None, norm: str = None, nor: str = None,
                  asinh: bool = False, vmin: float = None, vmax: float = None, cmap=cm.viridis):
        """Plot cached data loaded with ``load_data``."""
        if not hasattr(self, 'array'):
            raise ValueError('No cached data found. Run load_data(healpix_map) first.')
        norm = norm if norm is not None else nor
        vmin, vmax = self._resolve_vmin_vmax(self.array, vmin, vmax)

        self.figure = pyplot.figure() if isinstance(figure, type(None)) else figure
        self.axes = pyplot.subplot(111, projection=self.wcs, frame_class=EllipticalFrame) if isinstance(axes, type(None)) else axes
        self.img = self._imshow(self.array, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, asinh=asinh, norm_mode=norm)
        return self.img

    def remove_ticks(self):
        
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)
        
    def add_grid(self, color='k',prime_merid_linewidth=1.5, linewidths=1):
        """Add grid to image"""
        num_longitude_graticules = 12
        num_latitude_graticules = 5
        #self.axes.coords.grid(color=color, alpha=0.5, linestyle='dotted')
        #self.axes.coords['glon'].set_ticks(number=num_longitude_graticules)
        #self.axes.coords['glon'].set_ticklabel(color=color)
        ax = self.axes
        ax.coords.grid(color='black', linestyle='dotted', alpha=0.5, linewidth=linewidths)

        # Customize the prime meridian and equator lines
        #ax.coords[0].set_ticklabel(exclude_overlapping=True)
        #ax.coords[0].set_axislabel(exclude_overlapping=True)
        ax.coords[0].set_ticks(number=num_longitude_graticules)

        #ax.coords[1].set_ticklabel(exclude_overlapping=True)
        #ax.coords[1].set_axislabel(exclude_overlapping=True)
        ax.coords[1].set_ticks(number=num_latitude_graticules)

        # Plot the prime meridian and equator lines separately
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        prime_meridian = ax.plot([0, 0], [-90, 90], transform=ax.get_transform('world'), 
                                linestyle='solid', linewidth=prime_merid_linewidth, color='black', alpha=0.8)
        equator = ax.plot([-180, 180], [0, 0], transform=ax.get_transform('world'),
                        linestyle='solid', linewidth=prime_merid_linewidth, color='black', alpha=0.8)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def add_colorbar(self, unit_label=' ',  ticks=None):
        """Add colorbar"""
        
        axins1 = inset_axes(self.axes, width='60%', height='5%', loc='upper center', 
                    bbox_to_anchor=(0.,-1.,1,1), 
                    bbox_transform=self.axes.transAxes)
        cb = self.figure.colorbar(self.img,cax=axins1,orientation='horizontal',ticks=ticks)
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.xaxis.set_label_position('bottom')
        cb.set_label(unit_label)
        
    def add_loops(self):
        """ Add synchrotron loops to mollview plot""" 
        for k,v in loops.items():
            lons,lats = read_regions_file(v[0], regs_dir='../ancillary_data/loops/')
            lons, lats = greatcircleinterp(lons, lats, distance=0.1)
    
            self.axes.plot(lons,lats,transform=self.axes.get_transform('galactic'),lw=4,ls='--',color='k')
            self.axes.text(v[1][0],v[1][1],k, color='k', transform=self.axes.get_transform('galactic'),ha='center')


    def imshow(self, array,vmin=None,vmax=None,cmap=None,interpolation='nearest', norm='hist'):
        """Backward-compatible wrapper for imshow."""
        return self._imshow(array, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, norm_mode=norm)
    
    def add_contour(self, m,levels=[0.5,1],vmin=None,vmax=None,cmap=None,interpolation='nearest',linewidths=0.5,colors='k'):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        contour = self.axes.contour(array,colors=colors,levels=levels,vmin=vmin,vmax=vmax,linewidths=linewidths)
        return contour
    
    def contourf(self, m,levels=[0.5,1],vmin=None,vmax=None,cmap=None,interpolation='nearest'):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        axes_contour = pyplot.subplot(111,projection=self.wcs,frame_class=EllipticalFrame)

        
        contourf = axes_contour.contourf(array,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.5)
        array[np.isnan(array)] = 0
        contour = axes_contour.contour(array,colors='k',levels=levels,vmin=vmin,vmax=vmax,linewidths=0.5)
        return contourf 

    def norm(self, array, vmin, vmax, norm):
        """Normalise data""" 
        
        if isinstance(norm, type(None)):
            norm_module = None 
        else:
            if norm =='hist':
                amin = np.nanmin(array)
                st = HistEqStretch(array-np.nanmin(array))
                array = st(array-np.nanmin(array))
                norm_module=None
                if not isinstance(vmin,type(None)):
                    vmin = st(np.array([vmin-amin]))[0]
                if not isinstance(vmax,type(None)):
                    vmax = st(np.array([vmax-amin]))[0]
            else:
                norm_module = simple_norm(array,norm)
        return array, vmin, vmax, norm_module

    def add_title(self, text, **kwargs):
        """Wrapper for matplotlib title""" 
        self.axes.set_title(text,**kwargs)

    def text(self,x,y,text,**kwargs):
        """Wrapper for matplotlib text""" 
        self.axes.text(x,y,text,transform=self.axes.get_transform('galactic'),**kwargs)
                
                
@dataclass 
class Gnomview(_HealpixViewBase):
    
    map : HealpixMap = field(default_factory=lambda : np.zeros(1)) 
    wcs : WCS = field(default_factory=lambda : WCS(naxis=2)) 
    xwidth : float = 5
    ywidth : float = 5
    interpolation : str = 'nearest-neighbor' 
    
    crval : list = field(default_factory=lambda : [0,0]) 
    cdelt : list = field(default_factory=lambda : [-5./60.,5./60.])
    # Matplotlib info 
    axes : Axes = field(default_factory = lambda : None )
    figure : Figure = field(default_factory = lambda : None )

    projection : str = 'TAN'
    
    def __post_init__(self): 
        
        
        # build wcs first
        self.Nx,self.Ny = int(abs(self.xwidth//self.cdelt[0])), int(abs(self.ywidth//self.cdelt[1]))
        self.wcs.wcs.crpix=[self.Nx//2,self.Ny//2]
        self.wcs.wcs.cdelt=self.cdelt
        self.wcs.wcs.crval=self.crval
        self.wcs.wcs.ctype=[f'GLON-{self.projection}',f'GLAT-{self.projection}']

    @property
    def min_x(self):
        """Calculate the minimum x coordinate in degrees"""
        return self.crval[0] - self.xwidth/2
    @property
    def max_x(self):
        """Calculate the maximum x coordinate in degrees"""
        return self.crval[0] + self.xwidth/2
    
    @property
    def min_y(self):
        """Calculate the minimum y coordinate in degrees"""
        return self.crval[1] - self.ywidth/2
    @property
    def max_y(self):
        """Calculate the maximum y coordinate in degrees"""
        return self.crval[1] + self.ywidth/2
        
    def __call__(self, m : HealpixMap,
                 axes = None, 
                 figure = None, 
                 norm : str =None, 
                 asinh : bool = False,
                 vmin : float =None, vmax : float =None, cmap=cm.viridis): 
        """
        

        Parameters
        ----------
        m : HealpixMap
            DESCRIPTION.
        norm : str, optional
            DESCRIPTION. The default is None.
        vmin : float, optional
            DESCRIPTION. The default is None.
        vmax : float, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is cm.batlow_r.

        Returns
        -------
        None.

        """
        
        self.load_data(m)
        return self.plot_data(axes=axes, figure=figure, norm=norm, asinh=asinh, vmin=vmin, vmax=vmax, cmap=cmap)

    def load_data(self, healpix_map: HealpixMap):
        """Reproject and cache a HEALPix map for plotting."""
        self.array, self.footprint = self._reproject_healpix(healpix_map, self.wcs, [self.Ny, self.Nx], self.interpolation)
        return self.array

    def plot_data(self, axes=None, figure=None, norm: str = None, nor: str = None,
                  asinh: bool = False, vmin: float = None, vmax: float = None, cmap=cm.viridis):
        """Plot cached data loaded with ``load_data``."""
        if not hasattr(self, 'array'):
            raise ValueError('No cached data found. Run load_data(healpix_map) first.')
        norm = norm if norm is not None else nor
        vmin, vmax = self._resolve_vmin_vmax(self.array, vmin, vmax)

        self.figure = pyplot.figure() if isinstance(figure, type(None)) else figure
        self.axes = pyplot.subplot(111, projection=self.wcs) if isinstance(axes, type(None)) else axes

        if np.nansum(self.array) == 0:
            print('No data to plot')
            return

        self.img = self._imshow(self.array, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, asinh=asinh, norm_mode=norm)

        lon = self.axes.coords[0]
        lat = self.axes.coords[1]
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')

        return self.img
        
    def remove_ticks(self):
        
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)
    def remove_xaxis_ticks(self):
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)

    def remove_yaxis_ticks(self):
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)
        
    def add_grid(self, color='k'):
        """Add grid to image"""
        self.axes.coords.grid(color=color)
        self.axes.coords['glon'].set_ticklabel(color=color)

    def add_colorbar(self, unit_label=' ',  ticks=None):
        """Add colorbar"""
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=pyplot.Axes)
        if len(self.axes.images) > 0:
            cbar = self.figure.colorbar(self.axes.images[0], ax=self.axes, cax=cax,label=unit_label, ticks=ticks)

        #axins1 = inset_axes(self.axes, width='5%', height='100%', loc='upper center', 
        #            bbox_to_anchor=(0.6,0.,1,1), 
        #            bbox_transform=self.axes.transAxes, axes_class=pyplot.Axes)
        #cb = self.figure.colorbar(self.img,cax=axins1,orientation='vertical',ticks=ticks)
        #cb.ax.xaxis.set_ticks_position('bottom')
        #cb.ax.xaxis.set_label_position('bottom')
        #cb.set_label(unit_label)
            
    def add_overlay(self, lic, cmap='Greys', alpha=0.5,vmin=None,vmax=None):
        """Add overlay to image"""
        array, footprint = self._reproject_healpix(lic, self.wcs, [self.Ny, self.Nx], self.interpolation)

        print('PLOTTING OVERLAY')
        print(np.nansum(array),np.nanmax(array),np.nanmin(array))
        self.axes.imshow(array, cmap=cmap, alpha=alpha, origin='lower', vmin=vmin, vmax=vmax)

    def add_contour(self, m,levels=[0.5,1],vmin=None,vmax=None,cmap=None,interpolation='nearest',linewidths=0.5,colors='k'):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        contour = self.axes.contour(array,colors=colors,levels=levels,vmin=vmin,vmax=vmax,linewidths=linewidths)
        return contour

    def contourf(self, m,vmin=None,vmax=None,cmap=None,levels=[0,1],interpolation='nearest',alpha=0.5):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        axes_contour = pyplot.subplot(111,projection=self.wcs)

        
        contourf = axes_contour.contourf(array,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
        array[np.isnan(array)] = 0
        contour = axes_contour.contour(array,colors='k',levels=levels,vmin=vmin,vmax=vmax,linewidths=0.5)
        return contourf 


    def imshow(self, array,vmin=None,vmax=None,cmap=None,interpolation='nearest', norm='hist'):
        """Backward-compatible wrapper for imshow."""
        return self._imshow(array, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, norm_mode=norm)
    
    def norm(self, array, vmin, vmax, norm):
        """Normalise data""" 
        
        if isinstance(norm, type(None)):
            norm_module = None 
        else:
            if norm =='hist':
                amin = np.nanmin(array)
                st = HistEqStretch(array-np.nanmin(array))
                array = st(array-np.nanmin(array))
                norm_module=None
                if not isinstance(vmin,type(None)):
                    vmin = st(np.array([vmin-amin]))[0]
                if not isinstance(vmax,type(None)):
                    vmax = st(np.array([vmax-amin]))[0]
            else:
                norm_module = simple_norm(array,norm)
        return array, vmin, vmax, norm_module
    

@dataclass 
class Arcview(_HealpixViewBase):
    
    map : HealpixMap = field(default_factory=lambda : np.zeros(1)) 
    wcs : WCS = field(default_factory=lambda : WCS(naxis=2)) 
    Nx : int = 2160#256
    Ny : int = 2160#256
    interpolation : str = 'nearest-neighbor' 
    
    crval : list = field(default_factory=lambda : [0,90]) 
    cdelt : list = field(default_factory=lambda : [-5./60.,5./60.])
    # Matplotlib info 
    axes : Axes = field(default_factory = lambda : None )
    figure : Figure = field(default_factory = lambda : None )
    
    def __post_init__(self): 
        
        
        # build wcs first
    
        self.wcs.wcs.crpix=[self.Nx//2,self.Ny//2]
        self.wcs.wcs.cdelt=self.cdelt
        self.wcs.wcs.crval=self.crval
        self.wcs.wcs.ctype=['GLON-ZEA','GLAT-ZEA']
        
    def __call__(self, m : HealpixMap,
                 axes = None, 
                 figure = None, 
                 norm : str =None, 
                 asinh : bool = False,
                 vmin : float =None, vmax : float =None, cmap=cm.viridis): 
        """
        

        Parameters
        ----------
        m : HealpixMap
            DESCRIPTION.
        norm : str, optional
            DESCRIPTION. The default is None.
        vmin : float, optional
            DESCRIPTION. The default is None.
        vmax : float, optional
            DESCRIPTION. The default is None.
        cmap : TYPE, optional
            DESCRIPTION. The default is cm.batlow_r.

        Returns
        -------
        None.

        """
        
        self.load_data(m)
        return self.plot_data(axes=axes, figure=figure, norm=norm, asinh=asinh, vmin=vmin, vmax=vmax, cmap=cmap)

    def load_data(self, healpix_map: HealpixMap):
        """Reproject and cache a HEALPix map for plotting."""
        self.array, self.footprint = self._reproject_healpix(healpix_map, self.wcs, [self.Ny, self.Nx], self.interpolation)
        return self.array

    def plot_data(self, axes=None, figure=None, norm: str = None, nor: str = None,
                  asinh: bool = False, vmin: float = None, vmax: float = None, cmap=cm.viridis):
        """Plot cached data loaded with ``load_data``."""
        if not hasattr(self, 'array'):
            raise ValueError('No cached data found. Run load_data(healpix_map) first.')
        norm = norm if norm is not None else nor
        self.vmin, self.vmax = self._resolve_vmin_vmax(self.array, vmin, vmax)

        self.figure = pyplot.figure() if isinstance(figure, type(None)) else figure
        if isinstance(axes, type(None)):
            self.axes = pyplot.subplot(111, projection=self.wcs, frame_class=EllipticalFrame)
            self.axes.coords.frame.set_frame_shape('circle')
        else:
            self.axes = axes

        if np.nansum(self.array) == 0:
            print('No data to plot')
            return

        self.img = self._imshow(self.array, cmap=cmap, interpolation='nearest', vmin=self.vmin, vmax=self.vmax, asinh=asinh, norm_mode=norm)
        # Clip the image to the frame
        self.img.set_clip_path(self.axes.coords.frame.patch)

        lon = self.axes.coords[0]
        lat = self.axes.coords[1]
        #lon.set_ticks_visible(False)
        #lon.set_ticklabel_visible(False)
        #lat.set_ticks_visible(False)
        #lat.set_ticklabel_visible(False)
        #lat.set_ticks([1600]*units.degree)
        #lon.set_axislabel('Galactic Longitude')
        #lat.set_axislabel('Galactic Latitude')

        return self.img
        
    def remove_ticks(self):
        
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)
    def remove_yaxis_ticks(self):
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)

    def add_grid(self, color='k'):
        """Add grid to image"""
        self.axes.coords.grid(color=color)
        self.axes.coords[0].set_ticklabel(color=color)

    def add_colorbar(self, unit_label=' ',  ticks=None):
        """Add colorbar"""
        
        axins1 = inset_axes(self.axes, width='5%', height='100%', loc='upper center', 
                    bbox_to_anchor=(0.7,0.,1,1), 
                    bbox_transform=self.axes.transAxes)
        if hasattr(self, 'img'):
            cb = self.figure.colorbar(self.img,cax=axins1,orientation='vertical',ticks=ticks)
            #cb.ax.xaxis.set_ticks_position('bottom')
            #cb.ax.xaxis.set_label_position('bottom')
            cb.set_label(unit_label)

    def contourf(self, m,levels=[0.5,1],vmin=None,vmax=None,cmap=None,interpolation='nearest'):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        axes_contour = pyplot.subplot(111,projection=self.wcs,frame_class=EllipticalFrame)

        
        contourf = axes_contour.contourf(array,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.5)
        array[np.isnan(array)] = 0
        contour = axes_contour.contour(array,colors='k',levels=levels,vmin=vmin,vmax=vmax,linewidths=0.5)
        return contourf 


    def imshow(self, array,vmin=None,vmax=None,cmap=None,interpolation='nearest', norm='hist'):
        """Backward-compatible wrapper for imshow."""
        return self._imshow(array, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax, norm_mode=norm)


@dataclass
class Cartview(_HealpixViewBase):

    map: HealpixMap = field(default_factory=lambda: np.zeros(1))
    wcs: WCS = field(default_factory=lambda: WCS(naxis=2))
    xwidth: float = 360
    ywidth: float = 180
    interpolation: str = 'nearest-neighbor'

    crval: list = field(default_factory=lambda: [0, 0])
    cdelt: list = field(default_factory=lambda: [-1. / 6, 1. / 6])
    axes: Axes = field(default_factory=lambda: None)
    figure: Figure = field(default_factory=lambda: None)

    projection: str = 'CAR'

    def __post_init__(self):
        self.Nx, self.Ny = int(abs(self.xwidth // self.cdelt[0])), int(abs(self.ywidth // self.cdelt[1]))
        self.wcs.wcs.crpix = [self.Nx // 2, self.Ny // 2]
        self.wcs.wcs.cdelt = self.cdelt
        self.wcs.wcs.crval = self.crval
        self.wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']

    def __call__(self, m: HealpixMap, axes=None, figure=None, norm: str = None,
                 asinh: bool = False, vmin: float = None, vmax: float = None, cmap=cm.viridis):
        self.load_data(m)
        return self.plot_data(axes=axes, figure=figure, norm=norm, asinh=asinh, vmin=vmin, vmax=vmax, cmap=cmap)

    def load_data(self, healpix_map: HealpixMap):
        """Reproject and cache a HEALPix map for plotting."""
        self.array, self.footprint = self._reproject_healpix(healpix_map, self.wcs, [self.Ny, self.Nx], self.interpolation)
        return self.array

    def plot_data(self, axes=None, figure=None, norm: str = None, nor: str = None,
                  asinh: bool = False, vmin: float = None, vmax: float = None, cmap=cm.viridis):
        """Plot cached data loaded with ``load_data``."""
        if not hasattr(self, 'array'):
            raise ValueError('No cached data found. Run load_data(healpix_map) first.')
        norm = norm if norm is not None else nor
        vmin, vmax = self._resolve_vmin_vmax(self.array, vmin, vmax)

        self.figure = pyplot.figure() if isinstance(figure, type(None)) else figure
        self.axes = pyplot.subplot(111, projection=self.wcs) if isinstance(axes, type(None)) else axes

        if np.nansum(self.array) == 0:
            print('No data to plot')
            return

        self.img = self._imshow(self.array, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax, asinh=asinh, norm_mode=norm)

        lon = self.axes.coords[0]
        lat = self.axes.coords[1]
        lon.set_axislabel('Galactic Longitude')
        lat.set_axislabel('Galactic Latitude')

        return self.img

    def remove_ticks(self):
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)

    def remove_xaxis_ticks(self):
        self.axes.coords[0].set_ticks_visible(False)
        self.axes.coords[0].set_ticklabel_visible(False)

    def remove_yaxis_ticks(self):
        self.axes.coords[1].set_ticks_visible(False)
        self.axes.coords[1].set_ticklabel_visible(False)

    def add_grid(self, color='k'):
        """Add grid to image."""
        self.axes.coords.grid(color=color)
        self.axes.coords['glon'].set_ticklabel(color=color)

    def add_colorbar(self, unit_label=' ', ticks=None):
        """Add colorbar."""
        divider = make_axes_locatable(self.axes)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=pyplot.Axes)
        if len(self.axes.images) > 0:
            self.figure.colorbar(self.axes.images[0], ax=self.axes, cax=cax, label=unit_label, ticks=ticks)

    def add_overlay(self, lic, cmap='Greys', alpha=0.5, vmin=None, vmax=None):
        """Add map overlay to image."""
        array, footprint = self._reproject_healpix(lic, self.wcs, [self.Ny, self.Nx], self.interpolation)
        self.axes.imshow(array, cmap=cmap, alpha=alpha, origin='lower', vmin=vmin, vmax=vmax)

    def add_contour(self, m, levels=[0.5, 1], vmin=None, vmax=None, cmap=None,
                    interpolation='nearest', linewidths=0.5, colors='k'):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        contour = self.axes.contour(array, colors=colors, levels=levels, vmin=vmin, vmax=vmax, linewidths=linewidths)
        return contour

    def contourf(self, m, levels=[0.5, 1], vmin=None, vmax=None, cmap=None, interpolation='nearest'):
        array, footprint = self._reproject_healpix(m, self.wcs, [self.Ny, self.Nx], self.interpolation)
        contourf = self.axes.contourf(array, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)
        array[np.isnan(array)] = 0
        self.axes.contour(array, colors='k', levels=levels, vmin=vmin, vmax=vmax, linewidths=0.5)
        return contourf

    def norm(self, array, vmin, vmax, norm):
        """Normalise data""" 
        
        if isinstance(norm, type(None)):
            norm_module = None 
        else:
            if norm =='hist':
                amin = np.nanmin(array)
                st = HistEqStretch(array-np.nanmin(array))
                array = st(array-np.nanmin(array))
                norm_module=None
                if not isinstance(vmin,type(None)):
                    vmin = st(np.array([vmin-amin]))[0]
                if not isinstance(vmax,type(None)):
                    vmax = st(np.array([vmax-amin]))[0]
            else:
                norm_module = simple_norm(array,norm)
        return array, vmin, vmax, norm_module
    

class VectorPlotter:
    """
    Reads in a text file of format:
    # l,b 
    l0 b0
    l1 b1
    ... 
    """
    def __init__(self, ax, wcs,region_data):
        self.ax = ax
        self.region_data = region_data
        self.wcs = wcs

    def plot_regions(self, linestyle='--', color='black', linewidth=1,alpha=0.7):
        for region in self.region_data:
            # Extract region parameters from the dictionary
            txt_file = region['txt_file']
            linestyle = region.get('linestyle', '--')
            color = region.get('color', 'black')
            l_center = region.get('l_center', 0)
            b_center = region.get('b_center', 0)
            data = np.loadtxt(txt_file,delimiter=' ')

            # Create a Rectangle patch representing the region
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            for istart,iend in zip(range(0,data.shape[0]-1),range(1,data.shape[0])):
                self.ax.plot([data[istart,0],data[iend,0]],
                             [data[istart,1],data[iend,1]],
                             linestyle=linestyle,
                                color=color,
                             linewidth=linewidth,
                             transform=self.ax.get_transform('galactic'))
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)


            # Extract label parameters from the dictionary
            label = region.get('label', '')
            label_l = region.get('label_l', l_center)
            label_b = region.get('label_b', b_center)
            label_color = region.get('label_color', 'black')


            # Convert label coordinates to pixel coordinates using WCS
            label_coord = SkyCoord(l=label_l*u.deg, b=label_b*u.deg, frame='galactic')
            label_pixel = self.wcs.world_to_pixel(label_coord)

            # Add label to the region if provided
            if label:
                self.ax.text(label_pixel[0], label_pixel[1], label, color=label_color, fontsize=FONT_SIZE_NORMAL, ha='center', va='center')
