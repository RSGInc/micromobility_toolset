import numpy as np

from activitysim.core import inject
from activitysim.core.config import setting

from ..utils.io import load_trip_matrix, save_trip_matrix


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

        # read in trip tables
        base_trips = load_trip_matrix(segment, base=True)
        build_trips = load_trip_matrix(segment, base=False)

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
    print('writing to disk...')

    save_trip_matrix(user_ben, 'user_ben', col_names=['minutes'])
    save_trip_matrix(delta_trips, 'chg_trips')
    save_trip_matrix(delta_miles, 'chg_vmt', col_names=['value'])
    save_trip_matrix(delta_pollutants, 'chg_emissions', col_names=list(setting('pollutants').keys()))

    print('done.')

if __name__ == '__main__':
    benefits()
