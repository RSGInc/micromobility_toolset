import numpy as np

from activitysim.core import inject
from activitysim.core.config import (
    setting,
    data_file_path,
    read_model_settings)

from ..utils import (network, output)
from ..utils.input import read_taz_from_sqlite, read_matrix_from_sqlite


def benefits():
    # initialize configuration data
    trips_settings = read_model_settings('trips.yaml')

    # get number of zones to dimension matrices
    max_zone = setting('max_zone') + 1

    base_sqlite_file = data_file_path(setting('base_sqlite_file'))
    build_sqlite_file = data_file_path(setting('build_sqlite_file'))

    # read auto times and distances
    auto_skim = read_matrix_from_sqlite(
        base_sqlite_file, 'auto_skim', 'i', 'j')

    # initialize empty matrices
    delta_trips = np.zeros((max_zone, max_zone, len(trips_settings.get('modes'))))
    user_ben = np.zeros((max_zone, max_zone))

    # ignore np divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')

    print('')
    print('calculating vmt, emissions, and user benefits...')

    # loop over market segments
    for segment in trips_settings.get('segments'):

        table = segment + trips_settings.get('trip_table_suffix')

        # read in trip tables
        base_trips = read_matrix_from_sqlite(
            base_sqlite_file, table,
            trips_settings.get('trip_ataz_col'), trips_settings.get('trip_ptaz_col'))

        build_trips = read_matrix_from_sqlite(
            build_sqlite_file, table,
            trips_settings.get('trip_ataz_col'), trips_settings.get('trip_ptaz_col'))

        if base_trips.size == 0 or build_trips.size == 0:
            print('%s is empty or missing' % table)
            continue

        # calculate difference in trips
        delta_trips = delta_trips + build_trips - base_trips

        # calculate logsums
        base_logsum = np.log(1.0 + np.nan_to_num(base_trips[:, :, 6] /
                            (np.sum(base_trips, 2) - base_trips[:, :, 6])))
        build_logsum = np.log(1.0 + np.nan_to_num(build_trips[:, :, 6] /
                              (np.sum(build_trips, 2) - build_trips[:, :, 6])))

        # calculate user benefits
        user_ben = user_ben - np.sum(base_trips, 2) * \
            (build_logsum - base_logsum) / trips_settings.get('ivt_coef')[table]

    # calculate difference in vmt and vehicle minutes of travel
    delta_minutes = auto_skim[:, :, 0] * (delta_trips[:, :, 0] +
        delta_trips[:, :, 1] / 2.0 + delta_trips[:, :, 2] / setting('sr3_avg_occ'))
    delta_miles = auto_skim[:, :, 1] * (delta_trips[:, :, 0] +
        delta_trips[:, :, 1] / 2.0 + delta_trips[:, :, 2] / setting('sr3_avg_occ'))

    print('')
    print('User benefits (min.): ', int(np.sum(user_ben)))
    print('Change in bike trips: ', int(np.sum(delta_trips[:, :, 6])))
    print('Change in VMT: ', int(np.sum(delta_miles)))

    # calculate difference in pollutants
    delta_pollutants = np.zeros((max_zone, max_zone, len(setting('pollutants').keys())))
    for idx, pollutant in enumerate(setting('pollutants').items()):
        delta_pollutants[:, :, idx] = delta_miles * pollutant[1]['grams_per_mile'] + \
            delta_minutes * pollutant[1]['grams_per_minute']
        print('Change in g. ' + pollutant[0] + ': ', int(np.sum(delta_pollutants[:, :, idx])))

    print('')
    # print('writing to disk...')
    # output.write_matrix_to_sqlite(user_ben,resources.application_config.build_sqlite_file,'user_ben',['minutes'])
    # output.write_matrix_to_sqlite(delta_trips,resources.application_config.build_sqlite_file,'chg_trips',resources.mode_choice_config.modes)
    # output.write_matrix_to_sqlite(delta_miles,resources.application_config.build_sqlite_file,'chg_vmt',['value'])
    # output.write_matrix_to_sqlite(delta_pollutants,resources.application_config.build_sqlite_file,'chg_emissions',resources.application_config.pollutants)

if __name__ == '__main__':
    benefits()
