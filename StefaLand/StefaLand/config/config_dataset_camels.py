import os
import MFFormer
import pandas as pd
from types import SimpleNamespace
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

camels_basin_file = os.path.join('/storage/home/nrk5343/work/30.MFFormer/MFFormer/datasets/camels/531_basin_list.txt')
camels_station_ids = pd.read_csv(camels_basin_file, header=None, dtype=str).values[:, 0].tolist()

"""
'PRCP_nldas_extended', 'SRAD_nldas_extended', 'Tmax_nldas_extended','Tmin_nldas_extended', 'Vp_nldas_extended',
'prcp_maurer_extended','srad_maurer_extended', 'tmax_maurer_extended', 'tmin_maurer_extended','vp_maurer_extended', 
'prcp_daymet', 'srad_daymet', 'tmax_daymet', 'tmin_daymet','vp_daymet', 
'QObs'

"""
tmp = {
    'input_nc_file': os.path.join(server_dir, '/storage/home/nrk5343/work/MFFormer.Nicks/MFFormer.FineTune.V0.01/CAMELS_Frederik.nc'),
    'train_date_list': ["1999-10-01", "2008-09-30"],
    'val_date_list': ["1980-10-01", "1989-09-30"],
    'test_date_list': ["1989-10-01", "1999-09-30"],

    'time_series_variables': ['prcp_daymet', 'srad_daymet', 'tmax_daymet', 'tmin_daymet', 'vp_daymet', ],
    'target_variables': ['QObs'],
    'static_variables': ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max',
                         'gvf_diff', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                         'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac',
                         'geol_permeability', 'p_mean', 'pet_mean', 'aridity', 'frac_snow', 'high_prec_freq',
                         'high_prec_dur', 'low_prec_freq', 'low_prec_dur'],
    'static_variables_category': [],
    'station_ids': camels_station_ids,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        # 'weather': ['prcp_daymet', 'srad_daymet'],
        'topography': ['elev_mean', 'slope_mean'],
        'soil': ['sand_frac', 'silt_frac', 'clay_frac', 'soil_depth_pelletier'],
        'geology': ['geol_permeability', 'soil_porosity', 'soil_conductivity'],
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['area_gages2'],
    'negative_value_variables': [],
    'missing_time_series_variables': [],
}
config_dataset_camels = SimpleNamespace(**tmp)
