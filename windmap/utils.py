import os

import ntpath
import numpy as np
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader

import datetime
from datetime import timedelta, timezone

import shutil
import rasterio

import matplotlib
import matplotlib.pyplot as plt

#from mpl_toolkits.basemap import Basemap
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from windmap_streamline import Streamlines
from matplotlib.collections import LineCollection


def textplacer(axname,da_plt,pos_var,textstring,fontsize,horizontalalignment):
    """
    Places text over a plot

    Parameters
    ----------
    axname : string
    da_plt : variable
        matplotlib.pyplot as plt , da_plt = plt
    pos_var : dict
        containing x,y,width,height
    textstring : string
        text to be placed
    fontsize : int
    horizontalallignment : string
    left, right and center
    
    
    """
    xpos,ypos,width,height=pos_var['x'],pos_var['y'],pos_var['width'],pos_var['height']
    axname=da_plt.axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    axname.xaxis.set_ticks_position('none')
    axname.yaxis.set_ticks_position('none') 
    axname.set_xticklabels('')
    axname.set_yticklabels('')
    plt.text(0.5, 0.5,textstring, horizontalalignment=horizontalalignment,fontsize=fontsize,fontweight='bold',color='k', verticalalignment='center', transform =axname.transAxes)

    
    
def colorcoder(colorlist):
    """
    Converts sequence or string of colour code to tupple

    Parameters
    ----------
    colorlist : list
        list of colourcode in form of string or sequence
    
    Returns
    
    RGB tuple of three floats from 0-1

    """
    clconv = matplotlib.colors.ColorConverter().to_rgb
    color_code=[clconv(color) for color in colorlist]
    return color_code


def colorize(array, cmap):
    """
    Helper function for normalizing colourmap 

    Parameters
    ----------
    array : numpy array
        
    cmap: maptlotlib colour map
    
    Returns
    -------
    Normalized color map 
    """
    #normed_data = (array - array.min()) / (array.max() - array.min())
    cm = plt.cm.get_cmap(cmap)
    return cm(array)


def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    Colorbar helper functions, which create legends in the plot
    

    Parameters
    ----------
    ncolors: int
              number of colors in legend
    cmap: matplotlib color map obj
      
    
    Returns
    -------
    returns colorbar object to be used as legend in the plot
    """
    #cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable,orientation='horizontal',**kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar



def custom_legend_placer(cf_ncdata,da_plt,dataclasslist,dataclasslabel,legendpost,colorlist,legendtitle):
    """
    Helper function to place the legend and runs internal colour bar functions

    Parameters
    ----------
    cf_ncdata : int
        Description of arg1
    da_plt: matplotlib plt axes

    dataclasslist: legend cutoff value list

    dataclasslabel: legend label list

    legendpost: legend position descrption dict
    
    colorlist: list of colour list for legend and map ploting
    
    legendtitle: legend title

    Returns
    -------
    none

    """
    color_code=colorcoder(colorlist)
    legendpos=da_plt.axes([legendpost['x'],legendpost['y'],legendpost['width'],legendpost['height']],frame_on=False,zorder=10) 
    colormap = LinearSegmentedColormap.from_list("my_colormap",color_code, N=len(dataclasslist), gamma=1.0)
    class_pdata = np.digitize(cf_ncdata, dataclasslist)
    colored_data = colorize(class_pdata, colormap)
    cb = colorbar_index(ncolors=len(dataclasslabel), cmap=colormap,  labels=dataclasslabel,cax = legendpos);cb.ax.tick_params(labelsize=8)
    cb.ax.set_title(legendtitle,fontsize=8,fontweight='bold') 
    return colored_data, colormap


def fixed_locator_lon(x_min,x_max):
    no_of_grid=round(x_max-x_min)
    step=round(x_max-x_min)/no_of_grid
    locators=np.arange(x_min,x_max,step)
    return locators


def add_gridlines(mainmap,x_min,y_min,x_max,y_max):
    gl = mainmap.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.ylabels_left = False
    #gl.xlabels_bottom = False
    gl.xlabels_top = False
    #locators=fixed_locator_lon(x_min,x_max)
    gl.xlocator = mticker.FixedLocator([x_min+1,x_max-1])
    #gl.ylocator = mticker.FixedLocator([y_min-1,y_max+1])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}

def logoplacer(logofile,mainmap,pos_var):
    """
    Places logo over a plot.

    Parameters
    ----------
    logofile : file path
        path of the logo
    da_plt : variable
        matplotlib.pyplot as plt , da_plt = plt
    pos_var : dict
        containing x,y,width,height        

    
    Returns
    -------
    
        Description of return value

    """
    # #xpos,ypos,width,height=pos_var['x'],pos_var['y'],pos_var['width'],pos_var['height']
    # logo = inset_axes(mainmap,
    #                 width=1,                     # inch
    #                 height=1,                    # inch
    #                 bbox_transform=mainmap.transAxes, # relative axes coordinates
    #                 bbox_to_anchor=(0.5,0.5),    # relative axes coordinates
    #                 loc=3)    
    logo= mainmap.inset_axes([0.84, -0.05, 0.15, 0.15],frame_on=False,fc='None',alpha=0)
    #logo = inset_axes(da_plt, width="30%",  height="30%")
    #axins.imshow(Z2, extent=extent, interpolation="nearest",
    #      origin="lower")
    logofile = plt.imread(logofile)
    #logo=da_plt.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    #logo=fig.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    logo.xaxis.set_ticks_position('none')
    logo.yaxis.set_ticks_position('none') 
    logo.set_xticklabels('')
    logo.set_yticklabels('')
    logo.imshow(logofile)


def get_windspeed(u_pdata1,v_pdata1):
    """
    from : https://github.com/blaylockbk/Ute_WRF/blob/master/functions/wind_calcs.py
    Calculates the wind speed from the u and v wind components
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    U=u_pdata1
    V=v_pdata1
    WSPD = np.sqrt(np.square(U)+np.square(V))
    return WSPD



def wm_ws_map(params,windspeed,u_pdata1,v_pdata1,vds):
    """
    Use u and v vector to plot the wind stream line and 
    overlay with wind speed plot. There is a shape file
    is overlayed for to give int boundary 

    Parameters
    ----------
    params : parameter objects
        Object variable for folder name, file name etc
    windspeed : Numpy 2D array
        Wind speed generated from u and v compute, fliped to match the canvas
    u_pdata1 : Numpy 2D array
        Wind u vector, fliped to match the canvas
    v_pdata1 : TYPE
        Wind u vector, fliped to match the canvas
    vds : TYPE
        rasterio object to get extent and other maping descriptions

    Returns
    -------
    Generate map png file

    """
    x_min,x_max,y_min,y_max=vds.bounds[0],vds.bounds[2],vds.bounds[1],vds.bounds[3]
    mx = np.linspace(x_min, x_max, vds.width)
    my = np.linspace(y_min, y_max, vds.height)
    ncx, ncy = np.meshgrid(mx, my)
    print('do_windmap')
    #legendpost={'x':0.01,'y':0.03,'width':0.97,'height':0.03}
    dataclasslist=[4, 8, 14, 18, 22, 26]
    dataclasslabel=[4, 8, 14, 18, 22, 26]
    colorlist=['#20b2aa','#9acd32','#ffd700','#ff8c00','#ff0000','#800000','#330000']
    da_plt=plt
    da_plt.figure(figsize=(params.plotsize['width'],params.plotsize['height']))
    mainmap=da_plt.axes((params.mainmappost['x'],params.mainmappost['y'],params.mainmappost['width'],params.mainmappost['height']), projection=ccrs.PlateCarree(),zorder=10)
    #bg_shapes = list(shpreader.Reader(params.background_shpfile).geometries())
    json_file = "../static/ea_ghcf_simple.json"
    with open(json_file, "r") as f:
        geom = json.load(f)
    gdf = gp.GeoDataFrame.from_features(geom)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_geometries(
        gdf["geometry"], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="black"
    )
    add_gridlines(mainmap,x_min,y_min,x_max,y_max)
    mainmap.add_geometries(bg_shapes, ccrs.PlateCarree(),
    edgecolor='black', facecolor='none', alpha=1)
    formatted_dataclasslabel = [ '%.0f' % elem for elem in dataclasslabel ]
    colored_data, colormap=custom_legend_placer(windspeed,da_plt,dataclasslist,formatted_dataclasslabel,params.legendpost,colorlist,'Knot(nm/hr)')
    mainmap.imshow(colored_data, cmap=colormap,extent=(x_min,x_max,y_min,y_max))
    #mainmap.imshow(windspeed)
    s = Streamlines(ncx, ncy, u_pdata1, v_pdata1)
    for streamline in s.streamlines:
        x, y = streamline
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n = len(segments)
        D = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=-1))
        L = D.cumsum().reshape(n,1) + np.random.uniform(0,1)
        C = np.zeros((n,3))
        C[:] = (L*1.5) % 1
        #linewidths = np.zeros(n)
        #linewidths[:] = 1.5 - ((L.reshape(n)*1.5) % 1)
        # line = LineCollection(segments, color=colors, linewidth=linewidths)
        line = LineCollection(segments, color=C, linewidth=0.5)
        #lengths.append(L)
        #colors.append(C)
        #lines.append(line)
        mainmap.add_collection(line)
    textplacer('firstitle',da_plt,params.pos_firstitle,params.firstitle,10,'center')
    datefmt=params.startdate
    run=params.run
    textplacer('vartitle',da_plt,params.pos_vartitle,f'Wind speed in Knot(nm/hr), based on GFS {datefmt} run {run}',10,'center')
    textplacer('thirdtitle',da_plt,params.pos_thirdtitle,'On '+params.istdatefmt+ ' IST',10,'center')
    logoplacer(params.logofile,da_plt,params.logopos)
    output_png_file=params.localpath+'{}.png'.format(params.utcdatefmt)
    da_plt.savefig(output_png_file, transparent=False)
    da_plt.close()
    return output_png_file



