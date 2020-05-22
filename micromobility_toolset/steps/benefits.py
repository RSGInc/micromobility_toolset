import numpy as np

from ..model import step


@step()
def benefits(base_scenario, build_scenario):
    """
    This step compares walk, bike, and auto trips between the base and
    build scenarios and calculates differences in user benefits, emissions,
    trip count, and trip distance
    """
    nzones = base_scenario.num_zones
    assert nzones == build_scenario.num_zones

    # initialize empty matrices
    delta_trips = np.zeros((nzones, nzones, len(build_scenario.trip_settings.get('modes'))))
    user_ben = np.zeros((nzones, nzones))

    # ignore np divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')

    print("\ncalculating vmt, emissions, and user benefits...")

    # loop over market segments
    for segment in build_scenario.trip_settings.get('segments'):

        # read in trip tables
        base_trips = base_scenario.load_trip_matrix(segment)
        build_trips = build_scenario.load_trip_matrix(segment)

        # calculate difference in trips
        delta_trips = delta_trips + build_trips - base_trips

        base_bike_trips = np.sum(np.take(base_trips, base_scenario.bike_mode_indices, axis=2), 2)
        build_bike_trips = np.sum(np.take(build_trips, build_scenario.bike_mode_indices, axis=2), 2)

        # calculate logsums
        base_logsum = \
            np.log(1.0 + np.nan_to_num(
                base_bike_trips / (np.sum(base_trips, 2) - base_bike_trips)))

        build_logsum = \
            np.log(1.0 + np.nan_to_num(
                build_bike_trips / (np.sum(build_trips, 2) - build_bike_trips)))

        # calculate user benefits
        user_ben = user_ben - np.sum(base_trips, 2) * \
            (build_logsum - base_logsum) / build_scenario.trip_settings.get('ivt_coef')[segment]

    # calculate difference in vmt and vehicle minutes of travel
    #######################################
    # FIX: don't use hardcoded skim indexes
    #######################################
    occupancy_dict = base_scenario.trip_settings.get('occupancy')
    all_modes = base_scenario.trip_settings.get('modes')

    delta_minutes = base_scenario.auto_skim[:, :, 0]
    delta_miles = base_scenario.auto_skim[:, :, 1]
    for mode, denom in occupancy_dict.items():
        idx = all_modes.index(mode)
        delta_minutes = delta_minutes * (delta_trips[:, :, idx] / denom)
        delta_miles = delta_miles * (delta_trips[:, :, idx] / denom)


    print("\nUser benefits (min.): ", int(np.sum(user_ben)))
    print('Change in bike trips: ', int(np.sum(np.take(delta_trips, build_scenario.bike_mode_indices, axis=2))))
    print('Change in VMT: ', int(np.sum(delta_miles)))

    # calculate difference in pollutants
    pollutants_dict = build_scenario.trip_settings.get('pollutants')
    delta_pollutants = np.zeros((nzones, nzones, len(pollutants_dict.keys())))
    for idx, pollutant in enumerate(pollutants_dict.items()):
        delta_pollutants[:, :, idx] = delta_miles * pollutant[1]['grams_per_mile'] + \
            delta_minutes * pollutant[1]['grams_per_minute']
        print(f'Change in g. {pollutant[0]}: {int(np.sum(delta_pollutants[:, :, idx]))}')

    print("\nwriting results...")

    build_scenario.write_zone_matrix(
        user_ben,
        'user_ben.csv',
        col_names=['minutes'])

    build_scenario.write_zone_matrix(
        delta_trips,
        'chg_trips.csv',
        col_names=build_scenario.trip_settings.get('modes'))

    build_scenario.write_zone_matrix(
        delta_miles,
        'chg_vmt.csv',
        col_names=['value'])

    build_scenario.write_zone_matrix(
        delta_pollutants,
        'chg_emissions.csv',
        col_names=list(pollutants_dict.keys()))

    print('done.')
