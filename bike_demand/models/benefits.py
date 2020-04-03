import numpy as np

from activitysim.core import inject
from activitysim.core.config import setting

from ..utils.io import read_matrix


def benefits():
    # initialize configuration data
    trips_settings = inject.get_injectable('trips_settings')

    # get number of zones to dimension matrices
    nzones = inject.get_injectable('num_zones')

    # read auto times and distances
    auto_skim = inject.get_injectable('auto_skim')

    # initialize empty matrices
    delta_trips = np.zeros((nzones, nzones, len(trips_settings.get('modes'))))
    user_ben = np.zeros((nzones, nzones))

    # ignore np divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')

    print('')
    print('calculating vmt, emissions, and user benefits...')

    # loop over market segments
    for segment in trips_settings.get('segments'):

        table = segment + trips_settings.get('trip_table_suffix')

        # read in trip tables
        base_trips = read_matrix(table)
        build_trips = read_matrix(table)

        # calculate difference in trips
        delta_trips = delta_trips + build_trips - base_trips

        # calculate logsums
        ####################################
        # FIX: don't hard code these indices!
        #
        # use trip mode list
        ####################################
        base_logsum = np.log(1.0 + np.nan_to_num(base_trips[:, :, 6] /
                            (np.sum(base_trips, 2) - base_trips[:, :, 6])))
        build_logsum = np.log(1.0 + np.nan_to_num(build_trips[:, :, 6] /
                              (np.sum(build_trips, 2) - build_trips[:, :, 6])))

        # calculate user benefits
        user_ben = user_ben - np.sum(base_trips, 2) * \
            (build_logsum - base_logsum) / trips_settings.get('ivt_coef')[segment]

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
    delta_pollutants = np.zeros((nzones, nzones, len(setting('pollutants').keys())))
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
