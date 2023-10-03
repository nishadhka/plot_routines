import ntpath
import geopandas as gp
import mapclassify
import os

import mapclassify as mc
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.cm as cm
import numpy as np
from ast import literal_eval




def organize_data(params):
    """
    Creates a geodataframe with data value and shape
    polygon to be plotted

    Parameters
    ----------
    params : class object
        Input/Output parameter definitions.

    Returns
    -------
    geodf : GeoDataFrame
        geodataframe with data value and shape
    polygon to be plotted.

    """
    db=pd.read_csv(params.csvfile)
    db.columns=['COUNTRY','data']
    shppath='./'
    ea_cntr=gp.read_file(f'{shppath}Africa_Adm0_Country/Africa_Adm0_Country.shp')
    ea_cntr = ea_cntr.rename({'NA2_DESCRI': 'COUNTRY'}, axis=1)
    edf=pd.merge(ea_cntr,db,on='COUNTRY',how='left')
    edf['data']=edf['data'].fillna(0.0)
    mds1=edf[['COUNTRY','geometry','data']]
    mds1.columns=['cntrs','geometry','data']
    return mds1



def return_colormap(params):
    """
    Create colormap of matplotlib based on number of class and given colorcode

    Parameters
    ----------
    params : class object
        Input/Output parameter definitions.
        
    Returns
    -------
    c_cmap : Object
        matplotlib colormap.

    """
    c = matplotlib.colors.ColorConverter().to_rgb
    #color_code=[c("#006837"), c("#1a9850"), c("#66bd63"), c("#fdff00"), c("#ffe833"), c("#ffc566"), c("#ff9933"), c("#f46d43"), c("#d73027"), c("#a50026"), c("#660000")]
    color_code=[c("#40b5c4"), c("#225da8"), c("#081d58")]
    c_cmap = LinearSegmentedColormap.from_list("my_colormap",color_code, N=len(params.classif), gamma=1.0)
    return c_cmap


def get_axes(params):
    """
    

    Parameters
    ----------
    params : class object
        Input/Output parameter definitions.

    Returns
    -------
    da_plt : Matplotlib plot object
        
    mainmap : Matplotlib plot axes
        The mainmap is to be plotted on this axes.
    legend_pos : Matplotlib plot axes
        The legend bars and data ionterval values to be plotted on this axes.
    text_label_pos : Matplotlib plot axes
        Text lables are plotted on this axes, acts as layer over mainmap.

    """
    da_plt=plt
    da_plt.figure(figsize=(params.plotsize['width'],params.plotsize['height']))
    mainmap=da_plt.axes((params.mainmappost['x'],params.mainmappost['y'],params.mainmappost['width'],params.mainmappost['height']), projection=gcrs.PlateCarree(),zorder=10)
    legend_pos=da_plt.axes([params.legendpost['x'],params.legendpost['y'],params.legendpost['width'],params.legendpost['height']],frame_on=False,zorder=10)
    text_label_pos=da_plt.axes([params.textpost['x'],params.textpost['y'],params.textpost['width'],params.textpost['height']],frame_on=False,zorder=10)
    #inset_map_pos=da_plt.axes((params.inset_mappost['x'],params.inset_mappost['y'],params.inset_mappost['width'],params.inset_mappost['height']), projection=gcrs.PlateCarree(),zorder=10)
    return da_plt,mainmap,legend_pos,text_label_pos

def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    
    Parameters
    ----------
    ncolors : Integer
        Number of colours in the colorbar.
    cmap : colormap object
        Colormap object output from function return_colormap()
    labels : TYPE, optional
        DESCRIPTION. The default is None, data labels from function get_label().
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    colorbar : Matplotlib object
        The legend plotted object, an important function to make the bar color legends.

    """
    
    #cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable,**kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def get_label(params,geodf):
    """
    Classify the input data and generate data labels to be
    marked for the legend color boxes, 

    Parameters
    ----------
    params : class object
        Input/Output parameter definitions.
        
    geodf : GeoDataFrame
        geodataframe with data value and shape

    Returns
    -------
    plot_labels : Data labels object of mapclassify
        data labels from function get_label().

    """
    scheme = mc.UserDefined(geodf['data'], params.classif)
    E=scheme.bins;n=len(E);ns=n-1;Q=list(E);
    aa=geodf['data'].min()
    Q.insert(0,aa);S=Q[0:n]
    plot_labels = ["%.1f-%.1f" % (b, c) for b, c in zip(S,E)]
    return plot_labels



def legend_maker(params,legend_pos,geodf):
    """
    Genearte and position the legend based on 
    legend_pos

    Parameters
    ----------
    params : class object
        Input/Output parameter definitions.
    legend_pos : Matplotlib plot axes
        The legend bars and data ionterval values to be plotted on this axes.
    geodf : GeoDataFrame
        geodataframe with data value and shape

    Returns
    -------
    None.

    """
    plot_labels=get_label(params,geodf)
    cb = colorbar_index(ncolors=len(plot_labels), cmap=c_cmap,labels=plot_labels,cax = legend_pos);
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_title(params.legendtitle,fontsize=12,fontweight='bold') 
    return 






def text_label(text_label_pos,params):
    """
    

    Parameters
    ----------
    text_label_pos : Matplotlib plot axes
        Text lables are plotted on this axes, acts as layer over mainmap.
    params : class object
        Input/Output parameter definitions.

    Returns
    -------
    None.

    """
    text_label_pos.text(0.45,0.84, params.rp_text, fontsize=24,fontweight='bold',ha='center',va='center',color='k', transform =text_label_pos.transAxes)
    
  
def plot_map(geodf,da_plt,mainmap,c_cmap,params):
    """
    Major function which does the mainmap plotting of
    chloropleth using geodf dataframe. This also saves
    the final output into png file.

    Parameters
    ----------
    geodf : GeoDataFrame
        geodataframe with data value and shape
    polygon to be plotted.
    da_plt : Matplotlib plot object
    
    mainmap : Matplotlib plot axes
        The mainmap is to be plotted on this axes.
    cmap : colormap object
        Colormap object output from function return_colormap()
    params : class object
        Input/Output parameter definitions.

    Returns
    -------
    None.

    """
    #scheme = mc.UserDefined(geodf['data'], params.classif)
    scheme = mc.UserDefined(geodf['data'], params.classif)
    #scheme = mc.EqualInterval(geodf['data'], k=params.ngroup)
    gplt.choropleth(
    geodf, hue='data', projection=gcrs.PlateCarree(),
    edgecolor='#000000', linewidth=1,
    cmap=c_cmap,
    legend=False, scheme=scheme,ax=mainmap)  
    ######
    # gplt.choropleth(
    # , hue='data', projection=gcrs.PlateCarree(),
    # edgecolor='#000000', linewidth=.2,
    # cmap=c_cmap,
    # legend=False, scheme=scheme,ax=inset_map_pos)  
    flname=ntpath.splitext(ntpath.basename(params.csvfile))[0]
    output_png_file=f'{params.output_path}{flname}_{params.rp_year}.png'
    da_plt.savefig(output_png_file, transparent=False)
    da_plt.close()
    
    
def stitch_plots(fl_n1,fl_n2,fl_n3,params):
    """
    https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html

    Parameters
    ----------
    image_folder : TYPE
        DESCRIPTION.
    spi_prod : TYPE
        DESCRIPTION.
    lt_month : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    im1 = plt.imread(fl_n1)[:,:,:3]
    im2 = plt.imread(fl_n2)[:,:,:3]
    im3 = plt.imread(fl_n3)[:,:,:3]
    fig = plt.figure(figsize=(24, 9.5))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
    #grid[-1].remove()
    #grid[-2].remove()
    laxes=fig.add_axes([0.5,0.25, 0.3, 0.1], frame_on=False,zorder=0)
    laxes.xaxis.set_ticks_position('none')
    laxes.yaxis.set_ticks_position('none') 
    laxes.set_xticklabels('')
    laxes.set_yticklabels('')
    #legend_maker(laxes)
    for ax, im in zip(grid, [im1, im2, im3]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')
    #plt.show()
    #for imgno in range(0,10):
    os.remove(fl_n1)
    os.remove(fl_n2)
    os.remove(fl_n3)
    flname=ntpath.splitext(ntpath.basename(params.csvfile))[0]
    fl_n=f'{params.output_path}{flname}.png'
    plt.savefig(fl_n,bbox_inches='tight',dpi=100)


class bin_create_params:
    plotsize={'width':8.0,'height':9.5}
    mainmappost={'x':0.01,'y':-0.05,'width':1.05,'height':1.03}
    #inset_mappost={'x':0.6,'y':0.45,'width':0.45,'height':0.5}
    legendpost={'x':0.1,'y':0.02,'width':0.03,'height':0.25}
    textpost={'x':0.01,'y':0.01,'width':1.1,'height':1.1}
    classif=[ 10, 20, 30]
    input_path='./'
    csvfile=f'{input_path}chrips_kmj_101520_1980_2023_1d.csv'
    rp_year='rf_1'
    output_path=input_path
    output_filename_with_format='test.png'
    data_date='13Oct2021 Wednesday'
    legendtitle="No. of Participants"
    rp_text='Synergy Building Training'


#
params=bin_create_params()
params.csvfile='participants.csv'
geodf=organize_data(params)
#geodf=geodf['data'].fillna(0.0)

print(geodf)

da_plt,mainmap,legend_pos,text_label_pos=get_axes(params)
c_cmap=return_colormap(params)
legend_maker(params,legend_pos,geodf)
text_label(text_label_pos,params)
plot_map(geodf,da_plt,mainmap,c_cmap,params)
#%
