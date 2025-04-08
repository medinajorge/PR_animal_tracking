"""
This script is used to download environmental data to complement the geographical locations
"""
import cdsapi
import numpy as np
import pandas as pd
import subprocess
import gc
import os
from pathlib import Path
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell': # script being run in Jupyter notebook
        from tqdm.notebook import tqdm
    elif shell == 'TerminalInteractiveShell': #script being run in iPython terminal
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm # Probably running on standard Python interpreter

import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
from tidypath import storage
from julia import Main     # Move these two lines to add_numbers_from_julia() to call from thread
Main.include(os.path.join(RootDir, "utils/julia_utils.jl"))


import getopt
try:
    opts, args = getopt.getopt(sys.argv[1:], "y:",
                               ["year="]) # argv[0] is the name of the script
except getopt.GetoptError:
    print('test.py -y <year>')

for opt, arg in opts:
    if opt in ("-y", "--year"):
        years = [int(i) for i in arg.split(',')]
print('Retrieved data: \n years: {}'.format(years))

v2 = True

filenames = ["SingleLevels_%s.grib" % y for y in years]
parentDir = os.path.join(RootDir, 'utils/data/SingleLevels')
if v2:
    parentDir = os.path.join(parentDir, 'v2')
Path(parentDir).mkdir(exist_ok=True, parents=True)
os.chdir(parentDir)

int_parser = lambda i: f'0{i}' if i < 10 else str(i)
sign = lambda n: 0 if n == 0 else n / abs(n)
cd_formatter = lambda c, m: c + 0.25*sign(c) if abs(c + 0.25*sign(c)) < m else m*sign(c)

print('Loading full DataFrame')
data = pd.read_csv(os.path.join(RootDir, 'data/dataset.csv'))
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
data['Year'] = data['DATE_TIME'].dt.year

c = cdsapi.Client()
if v2:
    variables = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                 'mean_sea_level_pressure', 'sea_surface_temperature',
                 'mean_wave_direction', 'mean_wave_period', 'significant_height_of_combined_wind_waves_and_swell',
                'total_precipitation', 'k_index',
                 'sea_ice_cover', 'surface_net_solar_radiation',
                'surface_net_thermal_radiation',
                 'near_ir_albedo_for_diffuse_radiation', 'near_ir_albedo_for_direct_radiation',
                 'downward_uv_radiation_at_the_surface',
                 'evaporation',
                 'geopotential'
                 ]
else:
    variables = [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'ice_temperature_layer_1', 'mean_sea_level_pressure', 'mean_wave_direction',
            'mean_wave_period', 'sea_surface_temperature', 'significant_height_of_combined_wind_waves_and_swell',
            'surface_pressure',
            'total_precipitation', 'k_index', '100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_neutral_wind',
            '10m_v_component_of_neutral_wind', 'ice_temperature_layer_4',
            'sea_ice_cover', 'surface_net_solar_radiation',
            'surface_net_thermal_radiation',
        ]

# 'maximum_individual_wave_height'

def cds_downloader(variables, year, month, days, area, month_file):
    c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': variables,
        'year': f'{year}',
        'month': month,
        'day': days,
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': area,
        'format': 'grib',
    },
    month_file
    )
    return

pbar = tqdm(range(len(years)))
for filename, year in zip(filenames, years):
    print("Garbage collection: {}".format(gc.collect()))
    pbar.refresh()
    print(pbar.update())

    yearDir = os.path.join(parentDir, f'{year}')
    Path(yearDir).mkdir(exist_ok=True, parents=True)
    os.chdir(yearDir)

    data_year = data[data['Year'] == year]
    data_year['Month'] = data_year['DATE_TIME'].dt.month
    data_year['LONGITUDE'] = data_year['LONGITUDE'].astype(np.float64, copy=False)
    data_year['LATITUDE'] = data_year['LATITUDE'].astype(np.float64, copy=False)
    print(f'Data loaded for year {year}')

    months = [int_parser(m) for m in set(data_year['Month'])]
    var_files = [filename.replace('.', f'_var-{var}.') for var in variables]

    for var_file, var in zip(var_files, variables):
        print(f'\n \n  Var: {var} \n \n')

        if Path(var_file).exists():
            print(f"{var_file} already existed.")

        else:
            month_files = []
            checked_var_in_df = False
            for month in months:
                month_file = var_file.replace('.', f'_month-{month}.')
                month_files.append(month_file)
                if not Path(month_file).exists():
                    data_month = data_year[data_year['Month'] == int(month)]
                    days = [int_parser(d) for d in set(data_month['DATE_TIME'].dt.day)]
                    lat_min, lat_max = cd_formatter(data_month['LATITUDE'].min(), 90), cd_formatter(data_month['LATITUDE'].max(), 90)
                    lon_min, lon_max = cd_formatter(data_month['LONGITUDE'].min(), 180), cd_formatter(data_month['LONGITUDE'].max(), 180)
                    area = [lat_max, lon_min, lat_min, lon_max]

                    cds_downloader(var, year, month, days, area, month_file)

                if not checked_var_in_df:
                    if Main.is_col_in_df(month_file, Main.get_weather_df(year, v2=v2)):
                        print("Variable already added. Verified at first downloaded month.")
                        checked_var_in_df = True
                        for file in month_files:
                            try:
                                os.remove(file)
                            except:
                                continue
                        break

            if checked_var_in_df:
                continue

            print(' {0} \n {1} \n {0}'.format('-'*100 + '\n' + '-'*100, 'MONTH DOWNLOAD COMPLETED'))

            print('Merging monthly files ...')
            bash_command = "cat {} > {}".format(" ".join(month_files), var_file)
            subprocess.run(bash_command, shell=True, check=True)

            print('Month files merged in year file. Deleting month files')
            for file in month_files:
                os.remove(file)

            print('Done!')


        print('Adding variable to the dataframe')
        Main.complete_df(year, v2=v2)

        print(f'Variable {var} added. Deleting its {year} file.')
        os.remove(var_file)


print('Finished!')
