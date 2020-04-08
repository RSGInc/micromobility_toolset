import numpy as np

from activitysim.core.inject import step, get_injectable
from activitysim.core.config import setting

from ..utils.io import (
    load_skim,
    load_util_table,
    load_taz_matrix,
    save_taz_matrix)


@step()
def initial_demand():
    # initialize configuration data
    trips_settings = get_injectable('trips_settings')

    nzones = get_injectable('num_zones')

    walk_skim = load_skim('walk')
    bike_skim = load_skim('bike')

    np.seterr(divide='ignore', invalid='ignore')

    print('performing model calculations...')
    for segment in trips_settings.get('segments'):

        # read in trip tables
        base_trips = load_taz_matrix(segment, base=True)
        base_motor_util = load_util_table(segment)

        base_bike_util = bike_skim * trips_settings.get('bike_skim_coef')
        base_walk_util = walk_skim * trips_settings.get('walk_skim_coef')

        base_motor_util = base_motor_util * (np.ones((nzones, nzones)) - np.diag(np.ones(nzones)))

        base_bike_util = base_bike_util + trips_settings.get('bike_asc')[segment]
        base_walk_util = base_walk_util + trips_settings.get('walk_asc')[segment]
        base_bike_util = base_bike_util + trips_settings.get('bike_intrazonal')
        base_walk_util = base_walk_util + trips_settings.get('walk_intrazonal')

        bike_avail = (bike_skim > 0) + np.diag(np.ones(nzones))
        walk_avail = (walk_skim > 0) + np.diag(np.ones(nzones))

        base_bike_util = base_bike_util - 999 * (1 - bike_avail)
        base_walk_util = base_walk_util - 999 * (1 - walk_avail)

        midxs = get_injectable('motorized_mode_indices')
        widxs = get_injectable('walk_mode_indices')
        bidxs = get_injectable('bike_mode_indices')

        motorized_trips = np.sum(np.take(base_trips, midxs, axis=2), 2)
        bike_trips = np.sum(np.take(base_trips, bidxs, axis=2), 2)
        walk_trips = np.sum(np.take(base_trips, widxs, axis=2), 2)
        total_trips = motorized_trips + bike_trips + walk_trips

        print('')
        print(f'segment {segment}')
        print('initial trips')
        print('total motorized walk bike')
        print(int(np.sum(total_trips)),
              int(np.sum(motorized_trips)),
              int(np.sum(walk_trips)),
              int(np.sum(bike_trips)))

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

        save_taz_matrix(build_trips, segment)

        print('final trips')
        print('total motorized walk bike')
        print(int(np.sum(build_trips)),
              int(np.sum(build_motor_trips)),
              int(np.sum(build_walk_trips)),
              int(np.sum(build_bike_trips)))


if __name__ == '__main__':
    initial_demand()
