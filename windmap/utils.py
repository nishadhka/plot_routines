import os
import time
import ntpath
import moviepy.editor as mpy
import os
import glob
import ntpath
import pandas as pd
from datetime import datetime
from dateutil import tz
import numpy as np

import xarray as xr
import numpy as np
import geopandas as gp
import pandas as pd
import PseudoNetCDF as pnc

import metpy.calc as metpcalc
from metpy.units import units

from shapely.geometry import Polygon
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

from windmap_streamline import Streamlines

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



def foldercreator(path):
   if not os.path.exists(path):
        os.makedirs(path)



def filepath_provider(path):
    filepath=os.path.dirname(path)+'/'
    return filepath


def make_ist_time(nctime):
    ocdate=datetime.strptime(np.datetime_as_string(nctime, unit='m'), '%Y-%m-%dT%H:%M')
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('India/Kolkata') 
    utc_ocdate = ocdate.replace(tzinfo=from_zone)
    # Convert time zone
    ist_ocdate = utc_ocdate.astimezone(to_zone)
    return ist_ocdate.strftime('%Y-%m-%dT%H:%M')    

def make_ist_time_video(nctime):
    ocdate=datetime.strptime(np.datetime_as_string(nctime, unit='m'), '%Y-%m-%dT%H:%M')
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('India/Kolkata') 
    utc_ocdate = ocdate.replace(tzinfo=from_zone)
    # Convert time zone
    ist_ocdate = utc_ocdate.astimezone(to_zone)
    return ist_ocdate.strftime('%Y%m%d')    



def videomaker(pngdir, chosenfps,outputformat,outfilename):
    pngfiles=glob.glob(pngdir+'*.png')
    db=pd.DataFrame(pngfiles)
    db.columns=['fullpath']
    db['filename']=db['fullpath'].map(path_leaf)
    db['filenumber'] =db['filename'].apply(lambda x: x.split('.')[0])
    db['filenumber'] = db['filenumber'].astype(float)
    db1=db.sort_values('filenumber')
    sortedfilename=db1['fullpath'].tolist()
    output_base_name=pngdir+outfilename+'.'+outputformat
    clipdur=[1]*len(sortedfilename)
    clip = mpy.ImageSequenceClip(sortedfilename, durations=clipdur, load_images=True)
    if outputformat=='mp4':
       clip.write_videofile(output_base_name,audio=False,fps=chosenfps )
    else:
       clip.write_gif(output_base_name,program='ImageMagick',opt='optimizeplus',fuzz=1,fps=chosenfps )


def cf_standardizer(inncfile):
    outputfile='cf_'+path_leaf(inncfile)
    path_outputfile=filepath_provider(inncfile)+outputfile
    infile = pnc.pncopen(inncfile, addcf=True)
    infile.save(path_outputfile)

def citybound(dataframe,citycode):
    dataframe[['minx', 'miny', 'maxx', 'maxy']]=dataframe.geometry.bounds
    dataframe1=dataframe[dataframe['code']==citycode]
    dataframe2=dataframe1.reset_index()
    return dataframe2


def for_tempextract(ncfile,timestep):
    DS = xr.open_dataset(ncfile)
    da = DS.T2_K
    da2=da.isel(TSTEP=[timestep])
    pdata=da2.values
    pdata1=pdata[0,0,:,:]
    air_pdata = pdata1 - 273.15
    nctime=nctime=DS.time[timestep].values
    return air_pdata,nctime

def make_ncfiles_fortest():
   # Define the geographic boundaries
    latitude_bounds = (24, -11)  # North 24°, South -11°
    longitude_bounds = (-19, 53)  # West 19°, East 53°

# Generate latitude and longitude arrays
    latitudes = np.linspace(latitude_bounds[0], latitude_bounds[1], 100)
    longitudes = np.linspace(longitude_bounds[0], longitude_bounds[1], 100)

# Create a meshgrid for latitudes and longitudes
    lon, lat = np.meshgrid(longitudes, latitudes)

# Generate sample data for U10_MpS and V10_MpS
# Assuming simple sinusoidal variations for demonstration
    U10_MpS = np.sin(np.pi * lat / 180) * np.cos(np.pi * lon / 180)
    V10_MpS = np.cos(np.pi * lat / 180) * np.sin(np.pi * lon / 180)

# Create an xarray Dataset
    data = xr.Dataset(
        {
            "U10_MpS": (["latitude", "longitude"], U10_MpS),
            "V10_MpS": (["latitude", "longitude"], V10_MpS)
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes
        }
    )

# Add metadata to the dataset
    data.U10_MpS.attrs['units'] = 'm/s'
    data.U10_MpS.attrs['long_name'] = 'Eastward Wind Component at 10 Meters'
    data.V10_MpS.attrs['units'] = 'm/s'
    data.V10_MpS.attrs['long_name'] = 'Northward Wind Component at 10 Meters'
    data.attrs['description'] = 'Wind speed and direction data at 10 meters'
    data.attrs['title'] = 'Sample NetCDF for Wind Data'

# Save the dataset to a NetCDF file
    file_path = 'Sample_Wind_Data.nc'
    data.to_netcdf(path=file_path)



def for_windspeed(ncfile,timestep):
    DS = xr.open_dataset(ncfile)
    u_da = DS.U10_MpS
    u_da2=u_da.isel(TSTEP=[timestep])
    u_pdata=u_da2.values
    u_pdata1=u_pdata[0,0,:,:]
    v_da = DS.V10_MpS
    v_da2=v_da.isel(TSTEP=[timestep])
    v_pdata=v_da2.values
    v_pdata1=v_pdata[0,0,:,:]    
    u_pdata2=u_pdata1* units.meter / units.second
    v_pdata2=v_pdata1* units.meter / units.second
    windspeed=metpcalc.wind_speed(u_pdata2,v_pdata2)
    nctime=DS.time[timestep].values
    return windspeed,nctime


def for_windmap(ncfile,timestep):
    DS = xr.open_dataset(ncfile)
    u_da = DS.U10_MpS
    u_da2=u_da.isel(TSTEP=[timestep])
    u_pdata=u_da2.values
    u_pdata1=u_pdata[0,0,:,:]
    v_da = DS.V10_MpS
    v_da2=v_da.isel(TSTEP=[timestep])
    v_pdata=v_da2.values
    v_pdata1=v_pdata[0,0,:,:]    
    nctime=DS.time[timestep].values
    ncx=DS['longitude'].values
    ncy=DS['latitude'].values
    return ncx,ncy,u_pdata1,v_pdata1,nctime




def colorcoder(colorlist):
    clconv = matplotlib.colors.ColorConverter().to_rgb
    color_code=[clconv(color) for color in colorlist]
    return color_code


def colorize(array, cmap):
    normed_data = (array - array.min()) / (array.max() - array.min())
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data)


def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
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


def forecast_extent(plot_extent): 
    geo_fe= Polygon([[plot_extent['s'],plot_extent['w']], [plot_extent['e'],plot_extent['s']], [plot_extent['e'],plot_extent['n']], [plot_extent['w'],plot_extent['n']]])
    return geo_fe



def shapefile_patches(plotsize,mainmappost,cen_forecast_extent,plot_extent,shapefile):
    fig = plt.figure()
    fig.set_size_inches(plotsize['width'],plotsize['height'])
    mainmap=fig.add_axes([mainmappost['x'],mainmappost['y'],mainmappost['width'],mainmappost['height']],zorder=10) 
    #epsg=24374,lon_0=19.17,lat_0=72.98
    amap = Basemap(epsg=4326,lon_0=cen_forecast_extent.x,lat_0=cen_forecast_extent.y,ellps = 'WGS84',llcrnrlon=plot_extent['w'],\
               llcrnrlat=plot_extent['s'],urcrnrlon=plot_extent['e'],urcrnrlat=plot_extent['n'],lat_ts=0,resolution='i',area_thresh=1000.,suppress_ticks=False, ax=mainmap)
    shppath=filepath_provider(shapefile)
    os.chdir(shppath)
    shapflname=path_leaf(shapefile).split('.')[0]
    amap.readshapefile(shapflname,shapflname,color='none',zorder=2)
    df_map = pd.DataFrame({'geometry': [Polygon(xy) for xy in amap.India_state_code]})
    df_map['patches'] = df_map['geometry'].map(lambda x: PolygonPatch(x,facecolor='none', 
            edgecolor='white', lw=.5, alpha=1., zorder=4)) 
    return fig, df_map


def textplacer(axname,xpos,ypos,width,height,textstring):
    axname=fig.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    axname.xaxis.set_ticks_position('none')
    axname.yaxis.set_ticks_position('none') 
    axname.set_xticklabels('')
    axname.set_yticklabels('')
    plt.text(0.8, 0.5,textstring, horizontalalignment='right',fontsize=30,fontweight='bold',color='k', verticalalignment='center', transform =cityname.transAxes)


def textplacer(axname,fig,xpos,ypos,width,height,textstring,fontsize,horizontalalignment):
    axname=fig.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    axname.xaxis.set_ticks_position('none')
    axname.yaxis.set_ticks_position('none') 
    axname.set_xticklabels('')
    axname.set_yticklabels('')
    plt.text(0.5, 0.5,textstring, horizontalalignment=horizontalalignment,fontsize=fontsize,fontweight='bold',color='k', verticalalignment='center', transform =axname.transAxes)


def logoplacer(logofile,fig,xpos,ypos,width,height):
    logofile = plt.imread(logofile)
    logo=fig.add_axes([xpos,ypos,width,height], frame_on=False,zorder=0) 
    logo.xaxis.set_ticks_position('none')
    logo.yaxis.set_ticks_position('none') 
    logo.set_xticklabels('')
    logo.set_yticklabels('')
    logo.imshow(logofile)


def udlegendcreator(cf_ncdata,dataclasslist,fig,legendpost,colorlist,legendlabel,legendtitle):
    color_code=colorcoder(colorlist)
    #dataclasslist.insert(0, cf_ncdata.min())
    #dataclasslist.insert(len(dataclasslist), np.inf)
    legendpos=fig.add_axes([legendpost['x'],legendpost['y'],legendpost['width'],legendpost['height']],zorder=10) 
    colormap = LinearSegmentedColormap.from_list("my_colormap",color_code, N=len(dataclasslist), gamma=1.0)
    class_pdata = np.digitize(cf_ncdata, dataclasslist)
    colored_data = colorize(class_pdata, colormap)
    #jenks_labels=dataclasslist[1:-1]
    cb = colorbar_index(ncolors=len(legendlabel), cmap=colormap,  labels=legendlabel,cax = legendpos);cb.ax.tick_params(labelsize=8)
    cb.ax.set_title(legendtitle,fontsize=8,fontweight='bold') 
    return dataclasslist,colormap 



def mapploter(plot_extent,fig,cf_ncfile,timestep,plotsize,mainmappost,df_map,dataclasslist,legendpost,colorlist,legendlabel,variable,legendtitle):
    if variable=='surface temperature':
       cf_ncdata,nctime=for_tempextract(cf_ncfile,timestep)
    if variable=='wind speed':
       cf_ncdata,nctime=for_windspeed(cf_ncfile,timestep)
    fig.set_size_inches(plotsize['width'],plotsize['height'])
    classif,colormap=udlegendcreator(cf_ncdata,dataclasslist,fig,legendpost,colorlist,legendlabel,legendtitle)
    class_pdata = np.digitize(cf_ncdata, classif)
    colored_data = colorize(class_pdata, colormap)
    mainmap=fig.add_axes([mainmappost['x'],mainmappost['y'],mainmappost['width'],mainmappost['height']],zorder=10)
    mainmap.set_xlabel('Longitude', fontsize=14,fontweight='bold')
    mainmap.set_ylabel('Latitude', fontsize=14,fontweight='bold') 
    concmap=mainmap.imshow(np.flipud(colored_data), cmap=colormap,extent=(plot_extent['w'], plot_extent['e'], plot_extent['s'],plot_extent['n']),alpha=1,interpolation='bilinear')
    pc = PatchCollection(df_map['patches'], match_original=True)
    mainmap.add_collection(pc)
    return concmap, nctime
    #fig.savefig(outputpath+'{0}.png'.format(str(timestep)), transparent=False)


def windmap_ploter(plot_extent,fig,cf_ncfile,timestep,plotsize,mainmappost,df_map,dataclasslist,legendpost,colorlist,legendlabel,variable,legendtitle):
    cf_ncdata,nctime=for_windspeed(cf_ncfile,timestep)
    fig.set_size_inches(plotsize['width'],plotsize['height'])
    classif,colormap=udlegendcreator(cf_ncdata,dataclasslist,fig,legendpost,colorlist,legendlabel,legendtitle)
    class_pdata = np.digitize(cf_ncdata, classif)
    colored_data = colorize(class_pdata, colormap)
    mainmap=fig.add_axes([mainmappost['x'],mainmappost['y'],mainmappost['width'],mainmappost['height']],zorder=10)
    mainmap.set_xlabel('Longitude', fontsize=14,fontweight='bold')
    mainmap.set_ylabel('Latitude', fontsize=14,fontweight='bold') 
    concmap=mainmap.imshow(np.flipud(colored_data), cmap=colormap,extent=(plot_extent['w'], plot_extent['e'], plot_extent['s'],plot_extent['n']),alpha=1,interpolation='bilinear')
    pc = PatchCollection(df_map['patches'], match_original=True)
    mainmap.add_collection(pc)
    variable='wind direction'
    ncx,ncy,u_pdata1,v_pdata1,nctime=for_windmap(cf_ncfile,timestep)    
    #lengths = []
    #colors = []
    #lines = []
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
    return concmap, nctime
