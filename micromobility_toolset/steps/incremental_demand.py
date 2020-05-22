import numpy as np

from ..model import step


@step()
def incremental_demand(base_scenario, build_scenario):
    """
    This step uses trip tables found in the base directory to generate
    analogous tables for the build directory. It uses the networks from the
    base and build directories to compare utilities.
    """

    nzones = base_scenario.num_zones
    assert nzones == build_scenario.num_zones

    midxs = base_scenario.motorized_mode_indices
    widxs = base_scenario.walk_mode_indices
    bidxs = base_scenario.bike_mode_indices

    # don't report zero divide in np arrayes
    np.seterr(divide='ignore', invalid='ignore')

    print("\nperforming scenario calculations...")

    # loop over market segments
    for segment in base_scenario.trip_settings.get('segments'):

        # use trips from previous step, if present
        base_trips = base_scenario.load_trip_matrix(segment)

        # calculate bike utilities
        base_bike_util = base_scenario.bike_skim * base_scenario.trip_settings.get('bike_skim_coef')
        build_bike_util = build_scenario.bike_skim * build_scenario.trip_settings.get('bike_skim_coef')

        # if not nhb, average PA and AP bike utilities
        if segment != 'nhb':
            base_bike_util = 0.5 * (base_bike_util + np.transpose(base_bike_util))
            build_bike_util = 0.5 * (build_bike_util + np.transpose(build_bike_util))

        # create 0-1 availability matrices when skim > 0
        if segment != 'nhb':
            bike_avail = \
                (base_scenario.bike_skim > 0) * \
                np.transpose(base_scenario.bike_skim > 0) + \
                np.diag(np.ones(nzones))
        else:
            bike_avail = (base_scenario.bike_skim > 0) + np.diag(np.ones(nzones))

        # non-available gets extreme negative utility
        base_bike_util = bike_avail * base_bike_util - 999 * (1 - bike_avail)
        build_bike_util = bike_avail * build_bike_util - 999 * (1 - bike_avail)

        # split full trip matrix and sum up into motorized, nonmotorized, walk, bike, and total
        motorized_trips = np.sum(np.take(base_trips, midxs, axis=2), 2)
        bike_trips = np.sum(np.take(base_trips, bidxs, axis=2), 2)
        walk_trips = np.sum(np.take(base_trips, widxs, axis=2), 2)
        total_trips = motorized_trips + bike_trips + walk_trips

        print(f"\n{segment} base trips")
        print(f'motorized: {int(np.sum(motorized_trips))}')
        print(f'walk: {int(np.sum(walk_trips))}')
        print(f'bike: {int(np.sum(bike_trips))}')
        print(f'total: {int(np.sum(total_trips))}')

        # calculate logit denominator
        denom = motorized_trips + bike_trips * np.exp(build_bike_util - base_bike_util)

        # perform incremental logit
        build_motor_trips = total_trips * np.nan_to_num(motorized_trips / denom)

        build_walk_trips = total_trips * np.nan_to_num(walk_trips / denom)

        build_bike_trips = total_trips * \
            np.nan_to_num(bike_trips * np.exp(build_bike_util - base_bike_util) / denom)

        # combine into one trip matrix and proportionally scale motorized sub-modes
        build_trips = base_trips.copy()
        for motorized_idx in midxs:
            build_trips[:, :, motorized_idx] = \
                base_trips[:, :, motorized_idx] * \
                np.nan_to_num(build_motor_trips / motorized_trips)

        for bike_idx in bidxs:
            build_trips[:, :, bike_idx] = \
                base_trips[:, :, bike_idx] * \
                np.nan_to_num(build_bike_trips / bike_trips)

        for walk_idx in widxs:
            build_trips[:, :, walk_idx] = \
                base_trips[:, :, walk_idx] * \
                np.nan_to_num(build_walk_trips / walk_trips)

        build_scenario.save_trip_matrix(build_trips, segment)

        print(f"\n{segment} build trips")
        print(f'motorized: {int(np.sum(build_motor_trips))}')
        print(f'walk: {int(np.sum(build_walk_trips))}')
        print(f'bike: {int(np.sum(build_bike_trips))}')
        print(f'total: {int(np.sum(build_trips))}')
