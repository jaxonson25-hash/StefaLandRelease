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
    'input_nc_file': '/storage/group/cxs1024/default/nrk5343/StefaLandData/GLOBAL_HydroBasin_level8_seamless_0.1_degree_2006_2018.nc',
    'train_date_list': ["2000-01-01", "2002-12-31"],
    'val_date_list': ["2000-01-01", "2002-12-31"],
    'test_date_list': ["2000-01-01", "2002-12-31"],

    'time_series_variables': ["P", "SWd", "RelHum", "Tmax", "Tmin", ],  # "LWd", "PET","SpecHum", "Temp", "Wind"
    'target_variables': [],
    'static_variables': ['MSWEP_P', 'MSWX_PET', 'MSWX_Tmean', 'MSWX_SWd',
                         'GMTED_elevation', 'GMTED_slope', 'GMTED_aspectcosine', 'HWSD_clay', 'HWSD_sand', 'HWSD_silt',
                         'porosity', 'Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'soil_depth',
                         'MODIS_NDVI', 'grassland_fraction_2018', 'forest_fraction_1992', 'forest_fraction_2018',
                         'catchsize', 'population_density_2000_buffer', 'gdp_2000_buffer', 'intact_forest'], #'soil_erosion',
    # 'HWSD_gravel',
    'static_variables_category': [],  # 'landcover', 'lithology'
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        'climate':['MSWEP_P', 'MSWX_PET', 'MSWX_Tmean', 'MSWX_SWd'],
        'topography': ['GMTED_elevation', 'GMTED_slope', 'GMTED_aspectcosine', 'soil_depth'],  #
        'soil': ['HWSD_clay', 'HWSD_sand', 'HWSD_silt'],
        'geology': ['Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'porosity'],
        'vegetation': ['MODIS_NDVI', 'grassland_fraction_2018', 'forest_fraction_1992', 'forest_fraction_2018', ],
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize', 'population_density_2000_buffer', 'gdp_2000_buffer', 'intact_forest'],
    'negative_value_variables': ["Tmax", "Tmin", "MSWX_Tmean", "meanTa", "permeability", "permeability_permafrost",
                                 "Permeability_no_permafrost", "GMTED_aspectcosine", "GMTED_elevation", "MODIS_NDVI"],
}

tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids.values

config_dataset_global_streamflow_175373_2006_2018 = SimpleNamespace(**tmp)
