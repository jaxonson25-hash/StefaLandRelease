import os

cwd = os.path.dirname(os.path.abspath(__file__))


class LoadPath(object):

    def __init__(self):
        pass

    @staticmethod
    def shapefile(shp_name):
        shp_dict = {
            "CONUS_WGS84": "CONUS/CONUS_WGS84.shp",
            "CONUS_Sinusoidal": "CONUS/CONUS_Sinusoidal.shp",
            "GLOBAL_WGS84": "GLOBAL/ne_110m_land.shp",
            "Oregon_WGS84": "CONUS/Oregon_WGS84.shp",

            "Africa": "HydroBasin/hybas_af_lev01_v1c.shp",
            "Arctic": "HydroBasin/hybas_ar_lev01_v1c.shp",
            "Asia": "HydroBasin/hybas_as_lev01_v1c.shp",
            "Australia": "HydroBasin/hybas_au_lev01_v1c.shp",
            "Europe": "HydroBasin/hybas_eu_lev01_v1c.shp",
            "Greenland": "HydroBasin/hybas_gr_lev01_v1c.shp",
            "North_America": "HydroBasin/hybas_na_lev01_v1c.shp",
            "South_America": "HydroBasin/hybas_sa_lev01_v1c.shp",
            "Siberia": "HydroBasin/hybas_si_lev01_v1c.shp",
        }
        shp_path = os.path.join(cwd, "shapefiles", shp_dict[shp_name])
        return shp_path

    def grid(self, grid_name):
        grid_dict = {}
        grid_path = os.path.join(cwd, "grid", grid_dict[grid_name])
        return grid_path

    def ref_SRS(self, ref_name):
        return self.grid(grid_name=ref_name)

    def prj(self, prj_name):
        prj_dict = {}
        prj_path = os.path.join(cwd, "prj", prj_dict[prj_name])
        return prj_path

    @staticmethod
    def server(server_name=None):
        return ""
        server_dict = {
            "suntzu": "/data/jql6620",
            "DoE": "/global/cfs/cdirs/m2637/jql6620",
            "wukong": "/projects/mhpi/jql6620",
        }

        if server_name is None:
            for server_name, server_path in server_dict.items():
                if os.path.exists(server_path):
                    return server_path
            raise FileNotFoundError("No server path found.")

        return server_dict[server_name]

    @staticmethod
    def station_ids(var_name):
        shp_dict = {
            "CAMELS_531": "camels/531_basin_list.txt",
            "CAMELS_671": "camels/671_basin_list.txt",
            "CAMELS_CHEM_516": "camels/camels_chem_516_basin_list.txt",
            "GLOBAL_4229": "GLOBAL_streamflow/4229_basin_list.txt",
            "PUR1":"GLOBAL_streamflow/PUR1.txt",
            "PUR4":"GLOBAL_streamflow/PUR4.txt",
            "GMD" : "Soilmoisture/GMD_ids.txt",
            "GLOBAL_3434": "GLOBAL_streamflow/3434_basin_list.txt",
            "GLOBAL_3262": "GLOBAL_streamflow/3262_basin_list.txt",
            "GLOBAL_3248": "GLOBAL_streamflow/3248_basin_list.txt",
            "GLOBAL_6738":"GLOBAL_streamflow/6738_global.txt",
            "GLOBAL_additional": "GLOBAL_streamflow/HydroBasin_level8_without_Q_additional.txt",
            "GLOBAL_additional_1952": "GLOBAL_streamflow/HydroBasin_level_1952_basin_basin_list.txt",
            "GLOBAL_HydroBasin_Additional": "GLOBAL_streamflow/Additional_Amazon_Conga_Asian_list.txt",
            "GLOBAL_3434_3262_HydroBasin": "GLOBAL_streamflow/3434_3262_HydroBasin_basin_list.txt",
            "GLOBAL_3434_3248_additional": "GLOBAL_streamflow/3434_3248_HydroBasin_basin_list.txt",
            "GLOBAL_3434_3248_additional_1952_selected_4000": "GLOBAL_streamflow/from_3434_3248_1952_random_selected_4000_station_ids.txt",
        }
        shp_path = os.path.join(cwd, shp_dict[var_name])
        return shp_path


datasets_path = LoadPath()
