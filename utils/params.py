"""
Modify if needed
"""

import numpy as np
from collections import defaultdict

Taxas = sorted("Bears Birds Sharks Penguins Seals Turtles Whales Seals-Eared Manta Fish Sirenians".split(" "))

xlims_original = [-0.726227492145896, 1.3465708864992056]
xlims_to_origin = [-0.616622604816556, 0.8882048533879336]

stage_mapping = {"HIVERNAGE": "non-breeding: winter/hivernage",
                 "non-breeding: winter": "non-breeding: winter/hivernage",
                 "Internesting": "breeding: internesting",
                 "Migration": "non-breeding: migration",
                 "REPRO": "breeding: repro",
                 "breeding": "breeding: unknown",
                 "non-breeding": "non-breeding: unknown",
                }

interesting_cols = ["ID", "COMMON_NAME", "Taxa", "Class", "SEX", "DATABASE", "TAG", "NumberOfSatellites", "Length", "Mean_latitude", "Mean_longitude", "Mean_year"]
secondary_cols = ["Order", "Family", "SatelliteProg", "TAG_TYPE", "ResidualError", "Stage", "AGE", "BODY_LENGTH"]
totally_unimportant_cols = ["Colour"]
updated_cols = ['Cluster ID', 'Cluster ID confidence', 'Cluster ID confidence interval', 'Animals in dataset', 'Animals in dataset interval', 'Length interval', "Mean year interval"]
all_cols = interesting_cols + secondary_cols + totally_unimportant_cols + updated_cols
artificial_label_cols = ['COMMON_NAME', 'Taxa', 'Class', 'SEX', 'DATABASE', 'TAG', 'Order', 'Family', 'SatelliteProg', 'TAG_TYPE', 'Stage', 'AGE']

discarded_weather_cols = ["2 metre temperature",
                          '100 metre U wind component', '100 metre V wind component', 'Neutral wind at 10 m u-component', 'Neutral wind at 10 m v-component',
                          'Surface pressure'
                         ]

weather_col_selection = ["Bathymetry", 'Sea ice area fraction', 'Sea surface temperature', 'Surface net solar radiation', 'Surface net thermal radiation',
                         'Mean sea level pressure', 'Significant height of combined wind waves and swell']

weather_cols = dict(temperature = ['Sea ice area fraction', 'Sea surface temperature',
                                   'Surface net solar radiation', 'Surface net thermal radiation'
                                  ],
                    wind = ['10 metre U wind component', '10 metre V wind component','K index',
                            'Mean sea level pressure', "Total precipitation"
                           ] ,
                    waves = ['Mean wave period', 'Significant height of combined wind waves and swell',
                             'Mean wave direction_x', 'Mean wave direction_y'],
                    bathymetry = ["Bathymetry"]
)
weather_cols_idxs = {}
pointer = 0
for k, v in weather_cols.items():
    list_len = len(v)
    weather_cols_idxs[k] = np.arange(pointer, pointer + list_len)
    pointer += list_len

weather_cols.update(dict(all=[col for value in weather_cols.values() for col in value]))
weather_cols_idxs["all"] = np.arange(len(weather_cols["all"]))

location_prunning= {'Black-browed albatross': np.array([(-65,-65), (-65, -20), (0,-48), (0, -65)])}

stage_split_by = {"Black-browed albatross": dict(column='Stage', colvalue=['breeding', 'non-breeding', 'unknown'])}

stage_species_v2 = ['Audouins gull',
                    'Corys shearwater',
                    'Northern gannet',
                    'Black-browed albatross',
                    'Bullers albatross',
                    'Grey-headed albatross',
                    'Hawksbill turtle',
                    'Chinstrap penguin',
                    'Scopolis shearwater',
                    'Macaroni penguin',
                    'Wandering albatross',
                    'Red-tailed tropic bird',
                    'Sooty tern',
                    'Baraus petrel',
                    'Wedge-tailed shearwater',
                    'Black-footed albatross',
                    'White-tailed tropic bird']

stage_species_v2_multilabel = ['Black-browed albatross',
                               'Black-footed albatross',
                               'Bullers albatross',
                               'Chinstrap penguin',
                               'Corys shearwater',
                               'Grey-headed albatross',
                               'Macaroni penguin',
                               'Scopolis shearwater',
                               'Wandering albatross']

stage_species_v2_binary = ['Baraus petrel',
                           'Hawksbill turtle',
                           'Red-tailed tropic bird',
                           'Sooty tern',
                           'Wedge-tailed shearwater',
                           'White-tailed tropic bird']

stage_species_v2_one_class = ['Audouins gull',
                              'Northern gannet']

breeding_remaps = {'breeding: chick-rearing': 'breeding: post-guard',
                   'breeding: creche': 'breeding: brood-guard',
                   'breeding: internesting': 'breeding: pre-egg'}

pad_day_rate_to_maxlen = {None: 200, 6: 100, 24: 160, 2:44, 3: 60, 1:33}

species_100_animals = ['Audouins gull',
                       'Corys shearwater',
                       'Northern gannet',
                       'Little penguin',
                       'Black-browed albatross',
                       'Southern elephant seal',
                       'Bullers albatross',
                       'Grey-headed albatross',
                       'Australian sea lion',
                       'Macaroni penguin',
                       'Loggerhead turtle',
                       'Chinstrap penguin',
                       'Scopolis shearwater',
                       'Blue shark',
                       'Shortfin mako shark',
                       'Whale shark',
                       'Tiger shark',
                       'Leatherback turtle',
                       'Wandering albatross',
                       'Northern elephant seal',
                       'Humpback whale',
                       'Salmon shark',
                       'Short-finned pilot whale',
                       'Adelie penguin',
                       'King eider',
                       'Hawksbill turtle',
                       'White shark',
                       'California sea lion',
                       'Masked booby',
                       'Green turtle',
                       'Long-nosed fur seal',
                       'Blue whale',
                       'Black-footed albatross',
                       'Trindade petrel',
                       'Laysan albatross',
                       'Ringed seal']

feature_map = {'x': 'x',
               'y': 'y',
               'z': 'z',
               'sin t': 'sin t',
               'cos t': 'cos t',
               'vomecrtn_1m': 'Marine current (meridional)',
               'vozocrte_1m': 'Marine current (zonal)',
               'vosaline_1m': 'Salinity',
               '10 metre U wind component': 'Wind (U)',
               '10 metre V wind component': 'Wind (V)',
               'Mean sea level pressure': 'Pressure',
               'Sea surface temperature': 'Temperature',
               'Mean wave period': 'Wave period',
               'Significant height of combined wind waves and swell': 'Wave height',
               'Total precipitation': 'Precipitation',
               'K index': 'K index',
               'Sea ice area fraction': 'Sea ice fraction',
               'Surface net short-wave (solar) radiation': 'Solar radiation',
               'Surface net long-wave (thermal) radiation': 'Thermal radiation',
               'Near IR albedo for diffuse radiation': 'IR albedo (diffuse)',
               'Near IR albedo for direct radiation': 'IR albedo (direct)',
               'Downward UV radiation at the surface': 'UV radiation',
               'Evaporation': 'Evaporation',
               'Geopotential': 'Geopotential',
               'Mean wave direction_x': 'Wave direction (x)',
               'Mean wave direction_y': 'Wave direction (y)',
               'coast-d': 'Distance to coast',
               'bathymetry': 'Bathymetry',
               'SN': 'SN',
               'WE': 'WE',
               'Location': 'Location',
               'Time': 'Time',
               'vomecrtn_97m': 'Marine current (meridional) 97m',
               'vozocrte_97m': 'Marine current (zonal) 97m',
               'votemper_97m': 'Temperature 97m',
               'vosaline_97m': 'Salinity 97m',
               'vomecrtn_1516m': 'Marine current (meridional) 1516m',
               'vozocrte_1516m': 'Marine current (zonal) 1516m',
               'votemper_1046m': 'Temperature 1046m',
               'votemper_10m': 'Temperature 10m',
               'vosaline_10m': 'Salinity 10m',
               }

feature_importance_settings = {'common_origin': dict(weather='all', common_origin_distance=True),
                               'env': dict(weather='all', common_origin_distance=False),
                               # 'common_origin_no_env': dict(weather=None, common_origin_distance=True),
                               }

error_feature_map = {'SN delta': 'Latitudinal difference',
                      'WE delta': 'Longitudinal difference',
                      'SN extension': 'Latitudinal distance',
                      'WE extension': 'Longitudinal distance',
                      '# of observations': '# Observations in trajectory',
                      'median latitude': 'Latitude (median)',
                      'median longitude': 'Longitude (median)',
                      'mean year': 'Year',
                      # 'tracking period (mean)': 'Tracking period (mean)',
                      # 'tracking period (std)': 'Tracking period (std)',
                      'sampling frequency (mean)': 'Sampling frequency (mean)',
                      'sampling frequency (std)': 'Sampling frequency (std)',
                      'mean effort': 'Fishing effort (mean)',
                      'mean effort excluding NaNs': 'Fishing effort (mean, excluding NaNs)',
                      'trimmed mean effort': 'Fishing effort (trimmed mean)',
                      '25th percentile effort': 'Fishing effort (p25)',
                      '50th percentile effort': 'Fishing effort (median)',
                      '75th percentile effort': 'Fishing effort (p75)',
                      'max effort': 'Fishing effort (max)',
                      'effort std': 'Fishing effort (std)',
                      'mean biodiversity': 'Biodiversity (mean)',
                      'biodiversity std': 'Biodiversity (std)',
                      '25th percentile biodiversity': 'Biodiversity (p25)',
                      '50th percentile biodiversity': 'Biodiversity (median)',
                      '75th percentile biodiversity': 'Biodiversity (p75)',
                      'max biodiversity': 'Biodiversity (max)',
                      'mean SST anomaly': 'SST anomaly (mean)',
                      'SST anomaly std': 'SST anomaly (std)',
                      '25th percentile SST anomaly': 'SST anomaly (p25)',
                      '50th percentile SST anomaly': 'SST anomaly (median)',
                      '75th percentile SST anomaly': 'SST anomaly (p75)',
                      'max SST anomaly': 'SST anomaly (max)',
                      'counts-same-species (median)': 'Overlap<sub>SS</sub> (median)',
                      'counts-other-species (median)': 'Overlap<sub>DS</sub> (median)',
                      'counts-other-species-same-taxa (median)': 'Overlap<sub>DS,ST</sub> (median)',
                      'counts-ratio (median)': 'Overlap<sub>SS</sub> / Overlap<sub>DS</sub> (median)',
                      'counts-ratio-same-taxa (median)': 'Overlap<sub>SS</sub> / Overlap<sub>DS,ST</sub> (median)',
                      'counts-same-species (mean)': 'Overlap<sub>SS</sub> (mean)',
                      'counts-other-species (mean)': 'Overlap<sub>DS</sub> (mean)',
                      'counts-other-species-same-taxa (mean)': 'Overlap<sub>DS,ST</sub> (mean)',
                      'counts-ratio (mean)': 'Overlap<sub>SS</sub> / Overlap<sub>DS</sub> (mean)',
                      'counts-ratio-same-taxa (mean)': 'Overlap<sub>SS</sub> / Overlap<sub>DS,ST</sub> (mean)',
                      'N species': '# Trajectories (SS)',
                      'Birds': 'Birds',
                      'Cetaceans': 'Cetaceans',
                      'Fishes': 'Fishes',
                      'Penguins': 'Penguins',
                      'Polar bears': 'Polar bears',
                      'Seals': 'Seals',
                      'Sirenians': 'Sirenians',
                      'Turtles': 'Turtles',
                      'Argos': 'Tracking: Argos',
                      'Fastloc GPS': 'Tracking: Fastloc GPS',
                      'GLS': 'Tracking: GLS',
                      'GPS': 'Tracking: GPS',
                      'M': 'Sex: male',
                      'F': 'Sex: female',
                      'U': 'Sex: unknown',
                      'overlap-same/overlap-different': '⟨Overlap<sub>SS</sub>(median)⟩ / ⟨Overlap<sub>DS</sub>(median)⟩',
                      'overlap-same/overlap-different-same-taxa': '⟨Overlap<sub>SS</sub>(median)⟩ / ⟨Overlap<sub>DS,ST</sub>⟩(median)',
                      'Days in trajectory': 'Days in trajectory',
                      'Days in trajectory (year)': 'Days in trajectory (same year)'}

def html_subscript_to_latex(x):
    return x.replace("<sub>", "$_{").replace("</sub>", "}$").replace("⟨", "$\\langle$").replace("⟩", "$\\rangle$")
def get_error_feature_map_mpl():
    return {k: html_subscript_to_latex(v) for k, v in error_feature_map.items()}


species_to_nmin_fit = {'King eider': 5}
species_to_nmin_fit = defaultdict(lambda: None, species_to_nmin_fit)
species_to_nmax_fit = {}
species_to_nmax_fit = defaultdict(lambda: None, species_to_nmax_fit)

sample_size_for_acc_specs = {'env': dict(random_states=range(1, 6), common_origin_distance=False, weather='all', method='exponential'),
                             'common_origin': dict(random_states=range(1, 6), common_origin_distance=True, weather=None, method='exponential')}
                             # 'common_origin': dict(random_states=range(1, 4), common_origin_distance=True, weather=None, method='exponential')}
