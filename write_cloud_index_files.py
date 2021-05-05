
#! /home/scros/.conda/envs/myenv/bin/python

import sys
import os
sys.path.append('/bdd/pegase/CMV/codes/fonctions')
import transverse as tr
import datetime as dt
import numpy as np
import netCDF4 as nc
sys.path.append('/home/scros/SOFOG')
import cloud_index_functions as cal
import pandas as pd 
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')
from matplotlib.backends.backend_pdf import PdfPages
#from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import matplotlib.patches as patches
import matplotlib.axes as ax
import pvlib
from dateutil.relativedelta import *
import scipy.ndimage
from cftime import date2num
sys.path.append('/home/dbennour/DeepPV')
from cloud_index_functions import get_cloud_albedo, get_cmv


"""#!/usr/bin/env python3.6"""

class CAL_files(object):
    """Class containing functions for writing examples of netcdf files with cloud index"""

    def __init__(self):
        self.LR_large_size = 171 #Infrared resoluton
    
        self.LR_zoom_size = 50
        
        self.HR_large_size = 513 #Visible resolution
        self.HR_zoom_size = 150 
       
        self.sites = ['SIRTA','LISA','CDG']#,'NW','NE','SE','SW']
        self.coordinates = [(48.713,2.208),(48.8,2.4),(49,2.5)] #Coordinate of the 3 sites
        self.cartesian = []
        for i in self.coordinates:
            lat_lon = [i[0],i[1]] 
            self.cartesian.append(tr.lat_lon_to_pixel(lat_lon,'LR'))
        self.PATH_GROUND_ALBEDO = '/bdd/pegase/CMV/Sirta/'
        self.dirout = '/homedata/dbennour/DeepPV/CAL_CMV_slot/'
        self.imbricated_domains = [(1,1),(7,3),(23,13),(45,25),(89,49),(121,67),(171,95)] #size of the different domaines (h*l IR pixels)
        
        self.imbricated_domains_HR = [(21,9),(69,39),(135,75),(267,147),(363,134),(513,285)] #Equivalent for HR images (x3)

        
    


    def Build_DS_netcdf(self,day_start,day_end):
        """Compile DataArrays in a single xarray Dataset and save in netcdf"""
        #Create coordinate files
        #Geographical coordinate arrays initialisation
        lon = np.zeros((self.HR_large_size,self.HR_large_size),dtype = np.float32)
        lat = np.zeros((self.HR_large_size,self.HR_large_size),dtype = np.float32)
        
        #Center of MSG HRV image
        lat_lon = [self.coordinates[0][0],self.coordinates[0][1]]
        lig_col = tr.lat_lon_to_pixel(lat_lon,'HR')
        
        ymin = lig_col[0] - self.HR_large_size // 2
        ymax = lig_col[0] + self.HR_large_size // 2
        xmin = lig_col[1] - self.HR_large_size // 2
        xmax = lig_col[1] + self.HR_large_size // 2

        
        #Loops computing for each pixel : latitude,longitude
        x=0
        while x < self.HR_large_size:
            y=0
            while y < self.HR_large_size:
                lat_lon = tr.pixel_to_lat_lon([ymin + y ,xmin + x], 'HR')
            
                lat[y,x] = lat_lon[0]
                lon[y,x] = lat_lon[1]
                y += 1
            x += 1
            
 
        
#         CAL,time = cal.Build_CAL_DataArray(day_start,day_end,self.coordinates[0][0],self.coordinates[0][1],self.HR_large_size)
        CAL,time = cal.Build_CAL_DataArray(day_start,day_end,self.coordinates[0][0],self.coordinates[0][1],self.HR_large_size)
        CMV_x,CMV_y,cmv_time=cal.Build_CMV_DataArray(day_start,day_end,self.HR_large_size)
        
        time_units = 'minutes since 2016-01-01 00:00'
        time_vals = date2num(time, time_units)
    
        DS = xr.Dataset({

            'CAL': xr.DataArray(
                data = CAL,
                dims = ['time','x','y'],
                coords = {'time':('time', time_vals),'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'cloud_albedo','units':'-'}
            ),
#             'CMV_X': xr.DataArray(
#                 data = CMV_x,
#                 dims = ['cmv_time','x','y'],
#                 coords = {'cmv_time':('cmv_time', cmv_time),'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
#                 attrs= {'long_name':'CMV_X','units':'-'}
#             ),
#             'CMV_Y': xr.DataArray(
#                 data = CMV_y,
#                 dims = ['cmv_time','x','y'],
#                 coords = {'cmv_time':('cmv_time', cmv_time),'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
#                 attrs= {'long_name':'CMV_Y','units':'-'}
#             ),
            
            'time': xr.DataArray(
                data = time_vals,
                dims = ['time'],
                attrs= {'long_name':'time','units':'minutes since 2016-01-01 00:00'}#,'units':'hour since 1970-01-01 00:00:00', 'calendar':'proleptic_gregorian'}
            ),
            'lon': xr.DataArray(
                data = lon,
                dims = ['x','y'],
                attrs= {'long_name':'longitude','units':'decimal_degrees'}
            ),
            'lat': xr.DataArray(
                data = lat,
                dims = ['x','y'],
                attrs= {'long_name':'latitude','units':'decimal_degrees'}
            )
        }, attrs = {'Conventions':'CF-1.7','history':'Example created just for practice'}
        )
        
        
        fileout = self.dirout + 'CAL_'+'CMV_' + day_start + '-' + day_end + '.nc'
        DS.to_netcdf(fileout, encoding={'CAL':{'dtype':'uint8'}})

        
        

    def Build_DS_netcdf_year(self,year_start,year_end):
        """Compile DataArrays in a single xarray Dataset and save in netcdf"""
        #Create coordinate files
        #Geographical coordinate arrays initialisation
        lon = np.zeros((self.HR_large_size,self.HR_large_size),dtype = np.float32)
        lat = np.zeros((self.HR_large_size,self.HR_large_size),dtype = np.float32)
        
        #Center of MSG HRV image
        lat_lon = [self.coordinates[0][0],self.coordinates[0][1]]
        lig_col = tr.lat_lon_to_pixel(lat_lon,'HR')
        
        ymin = lig_col[0] - self.HR_large_size // 2
        ymax = lig_col[0] + self.HR_large_size // 2
        xmin = lig_col[1] - self.HR_large_size // 2
        xmax = lig_col[1] + self.HR_large_size // 2

        
        #Loops computing for each pixel : latitude,longitude
        x=0
        while x < self.HR_large_size:
            y=0
            while y < self.HR_large_size:
                lat_lon = tr.pixel_to_lat_lon([ymin + y ,xmin + x], 'HR')
            
                lat[y,x] = lat_lon[0]
                lon[y,x] = lat_lon[1]
                y += 1
            x += 1
            
 
        
#         CAL,time = cal.Build_CAL_DataArray(year_start,year_end,self.coordinates[0][0],self.coordinates[0][1],self.HR_large_size)
        CAL,time = cal.Build_CAL_DataArray_year(year_start,year_end,self.coordinates[0][0],self.coordinates[0][1],self.HR_large_size)
        CMV_X, CMV_Y,cmv_time= cal.Build_CMV_DataArray_year(year_start,year_end,self.HR_large_size)
        time_units = 'minutes since 2016-01-01 00:00'
        time_vals = date2num(time, time_units)
    
        DS = xr.Dataset({

            'CAL': xr.DataArray(
                data = CAL,
                dims = ['time','x','y'],
                coords = {'time':('time', time),'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'cloud_albedo','units':'-'}
            ),
            'CMV_X': xr.DataArray(
                data = CMV_X,
                dims = ['cmv_time','x','y'],
                coords = {'cmv_time':('cmv_time', cmv_time),'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'CMV_X','units':'-'}
            ),
            'CMV_Y': xr.DataArray(
                data = CMV_Y,
                dims = ['cmv_time','x','y'],
                coords = {'cmv_time':('cmv_time', cmv_time),'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'CMV_Y','units':'-'}
            ),  
                 
            
            'time': xr.DataArray(
                data = time,
                dims = ['time'],
                attrs= {'long_name':'time','units':'minutes since 2016-01-01 00:00'}#,'units':'hour since 1970-01-01 00:00:00', 'calendar':'proleptic_gregorian'}
            ),
            'lon': xr.DataArray(
                data = lon,
                dims = ['x','y'],
                attrs= {'long_name':'longitude','units':'decimal_degrees'}
            ),
            'lat': xr.DataArray(
                data = lat,
                dims = ['x','y'],
                attrs= {'long_name':'latitude','units':'decimal_degrees'}
            )
        }, attrs = {'Conventions':'CF-1.7','history':'Example created just for practice'}
        )
        
        fileout = self.dirout + 'CAL_CMV_' + year_start + '.nc'
#         DS.to_netcdf(fileout, encoding={'CAL':{'dtype':'uint8'}})
        DS.to_netcdf(fileout)

    
    
    
    def Build_DS_netcdf_slot(self,slot_start,slot_end):
        """Compile slot DataArrays in a single xarray Dataset and save in netcdf"""
        #Create coordinate files
        #Geographical coordinate arrays initialisation
        SStart = dt.datetime.strptime(slot_start,'%Y%m%d%H%M')
        SEnd = dt.datetime.strptime(slot_end,'%Y%m%d%H%M')
        lon = np.zeros((self.HR_large_size,self.HR_large_size),dtype = np.float32)
        lat = np.zeros((self.HR_large_size,self.HR_large_size),dtype = np.float32)
        
        #Center of MSG HRV image
        lat_lon = [self.coordinates[0][0],self.coordinates[0][1]]
        lig_col = tr.lat_lon_to_pixel(lat_lon,'HR')
        
        ymin = lig_col[0] - self.HR_large_size // 2
        ymax = lig_col[0] + self.HR_large_size // 2
        xmin = lig_col[1] - self.HR_large_size // 2
        xmax = lig_col[1] + self.HR_large_size // 2

        
        #Loops computing for each pixel : latitude,longitude
        x=0
        while x < self.HR_large_size:
            y=0
            while y < self.HR_large_size:
                lat_lon = tr.pixel_to_lat_lon([ymin + y ,xmin + x], 'HR')
            
                lat[y,x] = lat_lon[0]
                lon[y,x] = lat_lon[1]
                y += 1
            x += 1
            
 
        
#         CAL,time = cal.Build_CAL_DataArray(year_start,year_end,self.coordinates[0][0],self.coordinates[0][1],self.HR_large_size)
#         CAL,time = cal.Build_CAL_DataArray_year(year_start,year_end,self.coordinates[0][0],self.coordinates[0][1],self.HR_large_size)
#         CMV_X, CMV_Y,cmv_time= cal.Build_CMV_DataArray_year(year_start,year_end,self.HR_large_size)

        CAL_T0 = np.empty((self.HR_large_size,self.HR_large_size),dtype=np.byte)
        CAL_T015 = np.empty((self.HR_large_size,self.HR_large_size),dtype=np.byte)
        CMV_X= np.empty((self.HR_large_size,self.HR_large_size),dtype=np.int8)
        CMV_Y= np.empty((self.HR_large_size,self.HR_large_size),dtype=np.int8)
       
        if(np.isnan(get_cloud_albedo(SStart,self.HR_large_size)).all()==True):
            CAL_T0=255
        else:
            CAL_T0=get_cloud_albedo(SStart,self.HR_large_size)*100
            
        if(np.isnan(get_cloud_albedo(SEnd,self.HR_large_size)).all()==True):
            CAL_T015=255
        else:
            CAL_T015=get_cloud_albedo(SEnd,self.HR_large_size)*100
        
        CMV_X,CMV_Y= get_cmv(SStart,SEnd)
        
#         time_units = 'minutes since 2016-01-01 00:00'
#         time_vals = date2num(time, time_units)
    
        DS = xr.Dataset({

            'CAL_T0': xr.DataArray(
                data = CAL_T0,
                dims = ['x','y'],
                coords = {'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'cloud_albedo','units':'-'}
            ),
            'CAL_T0-15': xr.DataArray(
                data = CAL_T015,
                dims = ['x','y'],
                coords = {'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'cloud_albedo','units':'-'}
            ),
            'CMV_X': xr.DataArray(
                data = CMV_X,
                dims = ['x','y'],
                coords = {'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'CMV_X','units':'-'}
            ),
            'CMV_Y': xr.DataArray(
                data = CMV_Y,
                dims = ['x','y'],
                coords = {'lon':(['x','y'],lon),'lat':(['x','y'],lat)},
                attrs= {'long_name':'CMV_Y','units':'-'}
            ),  
                 
            
#             'time': xr.DataArray(
#                 data = time,
#                 dims = ['time'],
#                 attrs= {'long_name':'time','units':'minutes since 2016-01-01 00:00'}#,'units':'hour since 1970-01-01 00:00:00', 'calendar':'proleptic_gregorian'}
#             ),
            'lon': xr.DataArray(
                data = lon,
                dims = ['x','y'],
                attrs= {'long_name':'longitude','units':'decimal_degrees'}
            ),
            'lat': xr.DataArray(
                data = lat,
                dims = ['x','y'],
                attrs= {'long_name':'latitude','units':'decimal_degrees'}
            )
        }, attrs = {'Conventions':'CF-1.7','history':'Example created just for practice'}
        )
        
        fileout = self.dirout + 'CAL_CMV_' + slot_start+'_'+slot_end + '.nc'
#         DS.to_netcdf(fileout, encoding={'CAL':{'dtype':'uint8'}})
        DS.to_netcdf(fileout)

    

    def Read_CAL_DataArray(self,year_start,year_end,slot,x0,y0):
        """Function just here to control values written in netcdf files"""
        
        Slot = dt.datetime.strptime(slot,'%Y%m%d%H%M')
      
        filein = self.dirout  + variable + '_' + year_start + '-' + year_end + '.nc'
        variable = 'CAL' #CAL = Cloud Albedo 

        CAL = load_xr(filein,variable) #CAL est le DataArray contenant toutes les valeurs de Cloud Albedo dans le fichier
        #CAL est un tableau 3D (t,x,y) 
        
        Slot = dt.datetime.strptime(slot,'%Y%m%d%H%M') #Conversion du slot entré par l'utilisateur en objet DateTime
        CAL_selected_values = CAL.sel(time=[Slot],x=int(x0),y=int(y0))#CAL_selected_values est une valeur particulière de CAL
        # correspondant au temps définis par le slot, sur le pixel x0 et y0 entré par l'utilisateur
        #CAL_selected_values est de type DataArray mais sans dimensions
        
        print(CAL_selected_values.values)#Même si CAL_selected_values ne contient qu'une valeur, il s'agit toujours d'un DataArray
        #Il faut donc lui ajouter l'attribut .values pour faire afficher la valeur 

       

        #CAL contrôle
        
        CAL_controle = cal.get_cloud_albedo(Slot,self.HR_large_size)
        
        print('CAL',CAL_controle[int(x0),int(y0)])
        
def load_xr(file_path, var):
    DS = xr.open_dataset(file_path)
    return DS[var]



    

if __name__ == "__main__":
    c1 = CAL_files()
    c1.Build_DS_netcdf(sys.argv[1],sys.argv[2])
    
    #c1.Read_CAL_DataArray(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
    
