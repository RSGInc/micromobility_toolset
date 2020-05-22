import numpy as np

from ..model import step


@step()
def initial_demand(model):
    """
    Create initial trip tables weighted trip tables for the base scenario.

    Raw trips are read from the data directory and weighted by the base network
    utilities.

    Results will be placed in the base directory
    """

    nzones = model.num_zones
    trip_settings = model.trip_settings
    walk_skim = model.base_walk_skim
    bike_skim = model.base_bike_skim
    midxs = model.motorized_mode_indices
    widxs = model.walk_mode_indices
    bidxs = model.bike_mode_indices

    np.seterr(divide='ignore', invalid='ignore')

    print("\nperforming model calculations...")
    for segment in model.trip_settings.get('segments'):

        # read in trip tables
        base_trips = model.load_trip_matrix(segment)
        base_motor_util = model.load_util_matrix(segment)

        base_bike_util = bike_skim * trip_settings.get('bike_skim_coef')
        base_walk_util = walk_skim * trip_settings.get('walk_skim_coef')

        base_motor_util = base_motor_util * (np.ones((nzones, nzones)) - np.diag(np.ones(nzones)))

        base_bike_util = base_bike_util + trip_settings.get('bike_asc')[segment]
        base_walk_util = base_walk_util + trip_settings.get('walk_asc')[segment]
        base_bike_util = base_bike_util + trip_settings.get('bike_intrazonal')
        base_walk_util = base_walk_util + trip_settings.get('walk_intrazonal')

        bike_avail = (bike_skim > 0) + np.diag(np.ones(nzones))
        walk_avail = (walk_skim > 0) + np.diag(np.ones(nzones))

        base_bike_util = base_bike_util - 999 * (1 - bike_avail)
        base_walk_util = base_walk_util - 999 * (1 - walk_avail)

        motorized_trips = np.sum(np.take(base_trips, midxs, axis=2), 2)
        bike_trips = np.sum(np.take(base_trips, bidxs, axis=2), 2)
        walk_trips = np.sum(np.take(base_trips, widxs, axis=2), 2)
        total_trips = motorized_trips + bike_trips + walk_trips

        print(f"\n{segment} initial trips")
        print(f'motorized: {int(np.sum(motorized_trips))}')
        print(f'walk: {int(np.sum(walk_trips))}')
        print(f'bike: {int(np.sum(bike_trips))}')
        print(f'total: {int(np.sum(total_trips))}')

        denom = np.exp(base_motor_util) + np.exp(base_walk_util) + np.exp(base_bike_util)
        build_motor_trips = total_trips * np.nan_to_num(np.exp(base_motor_util) / denom)
        build_walk_trips = total_trips * np.nan_to_num(np.exp(base_walk_util) / denom)
        build_bike_trips = total_trips * np.nan_to_num(np.exp(base_bike_util) / denom)

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

        model.save_trip_matrix(build_trips, segment, 'base')

        print(f"\n{segment} final trips")
        print(f'motorized: {int(np.sum(build_motor_trips))}')
        print(f'walk: {int(np.sum(build_walk_trips))}')
        print(f'bike: {int(np.sum(build_bike_trips))}')
        print(f'total: {int(np.sum(build_trips))}')
