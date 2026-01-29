import os
import MFFormer
import pandas as pd
from types import SimpleNamespace
import xarray as xr
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

tmp = {
    'input_nc_file': os.path.join("/storage/home/nrk5343/scratch/Data/GLOBAL_6738.nc"),
    'train_date_list': ["1980-01-01", "2018-12-31"],
    'val_date_list': ["1998-01-01", "1998-12-31"],
    'test_date_list': ["1980-01-01", "1997-12-31"],
    
    'time_series_variables': ['P', 'Temp', 'PET', 'streamflow'],
    'target_variables': [],
    
    'static_variables': ['lat', 'lon', 'aridity', 'meanP', 'ETPOT_Hargr', 'NDVI', 'FW', 
                        'meanslope', 'SoilGrids1km_sand', 'SoilGrids1km_clay', 'SoilGrids1km_silt', 
                        'glaciers', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt', 
                        'meanelevation', 'meanTa', 'permafrost', 'permeability', 'seasonality_P', 
                        'seasonality_PET', 'snow_fraction', 'snowfall_fraction', 'catchsize', 
                        'Porosity'],
    'static_variables_category': [],
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        'topography': ['meanelevation', 'meanslope'],
        'soil': ['HWSD_clay', 'HWSD_sand', 'HWSD_silt', 'HWSD_gravel', 'SoilGrids1km_sand', 'SoilGrids1km_clay', 'SoilGrids1km_silt'],
        'geology': ['permeability', 'Porosity', 'glaciers', 'permafrost'],
        'vegetation': ['NDVI', 'FW'],
        'climate': ['aridity', 'meanP', 'ETPOT_Hargr', 'meanTa', 'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction']
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize'],
    'negative_value_variables': ["Temp", "meanTa", "permeability", "meanelevation"],
}

tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids.values

config_dataset_global_6738 = SimpleNamespace(**tmp)