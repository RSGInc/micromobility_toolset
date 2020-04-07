import numpy as np

from activitysim.core.inject import step, get_injectable

from ..utils.io import (
    load_skim,
    load_taz_matrix,
    save_taz_matrix)


@step()
def benefits():
    # initialize configuration data
    trips_settings = get_injectable('trips_settings')

    # get number of zones to dimension matrices
    nzones = get_injectable('num_zones')

    # read auto times and distances
    auto_skim = load_skim('auto')

    # get matrix indices for bike modes
    bidxs = get_injectable('bike_mode_indices')

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
        base_trips = load_taz_matrix(segment, base=True)
        build_trips = load_taz_matrix(segment, base=False)

        # calculate difference in trips
        delta_trips = delta_trips + build_trips - base_trips

        base_bike_trips = np.sum(np.take(base_trips, bidxs, axis=2), 2)
        build_bike_trips = np.sum(np.take(build_trips, bidxs, axis=2), 2)

        # calculate logsums
        base_logsum = np.log(1.0 +
                             np.nan_to_num(base_bike_trips /
                                           (np.sum(base_trips, 2) - base_bike_trips)))
        build_logsum = np.log(1.0 +
                              np.nan_to_num(build_bike_trips /
                                            (np.sum(build_trips, 2) - build_bike_trips)))

        # calculate user benefits
        user_ben = user_ben - np.sum(base_trips, 2) * \
            (build_logsum - base_logsum) / trips_settings.get('ivt_coef')[segment]

    # calculate difference in vmt and vehicle minutes of travel
    #######################################
    # FIX: don't use hardcoded skim indexes
    #######################################
    occupancy_dict = trips_settings.get('occupancy')
    all_modes = trips_settings.get('modes')

    delta_minutes = auto_skim[:, :, 0]
    delta_miles = auto_skim[:, :, 1]
    for mode, denom in occupancy_dict.items():
        idx = all_modes.index(mode)
        delta_minutes = delta_minutes * (delta_trips[:, :, idx] / denom)
        delta_miles = delta_miles * (delta_trips[:, :, idx] / denom)



    print('')
    print('User benefits (min.): ', int(np.sum(user_ben)))
    print('Change in bike trips: ', int(np.sum(np.take(delta_trips, bidxs, axis=2))))
    print('Change in VMT: ', int(np.sum(delta_miles)))

    # calculate difference in pollutants
    pollutants_dict = trips_settings.get('pollutants')
    delta_pollutants = np.zeros((nzones, nzones, len(pollutants_dict.keys())))
    for idx, pollutant in enumerate(pollutants_dict.items()):
        delta_pollutants[:, :, idx] = delta_miles * pollutant[1]['grams_per_mile'] + \
            delta_minutes * pollutant[1]['grams_per_minute']
        print('Change in g. ' + pollutant[0] + ': ', int(np.sum(delta_pollutants[:, :, idx])))

    print('')
    print('writing results...')

    save_taz_matrix(user_ben, 'user_ben', col_names=['minutes'])
    save_taz_matrix(delta_trips, 'chg_trips')
    save_taz_matrix(delta_miles, 'chg_vmt', col_names=['value'])
    save_taz_matrix(delta_pollutants, 'chg_emissions', col_names=list(pollutants_dict.keys()))

    print('done.')

if __name__ == '__main__':
    benefits()
