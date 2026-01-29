"""
errors or too much NaN: 'aridity', 'permafrost',
remove: 'glaciers',
"""
import os
import MFFormer
import pandas as pd
from types import SimpleNamespace
import xarray as xr
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

tmp = {
    'input_nc_file': '/storage/group/cxs1024/default/nrk5343/StefaLandData/GLOBAL_3434_3248_additional_1952_static_0.1_degree.nc',
    'train_date_list': ["2000-01-01", "2002-12-31"],
    'val_date_list': ["2000-01-01", "2002-12-31"],
    'test_date_list': ["2000-01-01", "2002-12-31"],

    'time_series_variables': ["P", "SWd", "RelHum", "Tmax", "Tmin", ],
    'target_variables': [],
    'static_variables': [
            'meanP', 'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction', 'meanTa',
            'meanelevation', 'meanslope', 'aspectcosine',
            'HWSD_clay', 'HWSD_sand', 'HWSD_silt', 
            'porosity', 'Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'soil_depth',
            'NDVI', 'grassland_fraction_2018', 'forest_fraction_1992', 'forest_fraction_2018',
            'catchsize', 'population_density_2000_buffer', 'gdp_2000_buffer', 'intact_forest'
        ],

    'static_variables_category': [],
    'station_ids': None,
    'regions': [],
    'add_coords': False,

    'group_mask_dict': {
        'climate':     ['meanP', 'seasonality_P', 'seasonality_PET', 'meanTa'],
        'topography':  ['meanelevation', 'meanslope', 'aspectcosine', 'soil_depth'],
        'soil':        ['HWSD_clay', 'HWSD_sand', 'HWSD_silt'],
        'geology':     ['Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'porosity'],
        'vegetation':  ['NDVI', 'grassland_fraction_2018', 'forest_fraction_1992', 'forest_fraction_2018'],
    },

    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize', 'population_density_2000_buffer', 'gdp_2000_buffer', 'intact_forest'],

    # keep names consistent with what's really present so they won't be clipped to 0
    'negative_value_variables': [
        "Tmax", "Tmin", "meanTa",
        "Permeability_no_permafrost", "aspectcosine", "meanelevation"
    ],
}


tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids

config_dataset_global_3434_3248_additional_1952 = SimpleNamespace(**tmp)