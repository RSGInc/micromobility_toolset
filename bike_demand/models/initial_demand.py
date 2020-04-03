import numpy as np

from activitysim.core import inject
from activitysim.core.config import setting

from ..utils.io import read_matrix


def initial_demand():
    # initialize configuration data
    trips_settings = inject.get_injectable('trips_settings')

    nzones = inject.get_injectable('num_zones')

    walk_skim = inject.get_injectable('walk_skim')
    bike_skim = inject.get_injectable('bike_skim')

    np.seterr(divide='ignore', invalid='ignore')

    print('performing model calculations...')
    for segment in trips_settings.get('segments'):

        trip_table = segment + trips_settings.get('trip_table_suffix')
        motutil_table = segment + trips_settings.get('motorized_util_table_suffix')

        # read in trip tables
        base_trips = read_matrix(trip_table)
        base_motor_util = read_matrix(motutil_table)

        base_bike_util = bike_skim * trips_settings.get('bike_skim_coef')
        base_walk_util = walk_skim * trips_settings.get('walk_skim_coef')

        base_motor_util = base_motor_util * (np.ones((nzones, nzones)) -
                                             np.diag(np.ones(nzones)))

        base_bike_util = base_bike_util + trips_settings.get('bike_asc')[segment]
        base_walk_util = base_walk_util + trips_settings.get('walk_asc')[segment]
        base_bike_util = base_bike_util + trips_settings.get('bike_intrazonal')
        base_walk_util = base_walk_util + trips_settings.get('walk_intrazonal')

        bike_avail = (bike_skim > 0) + np.diag(np.ones(nzones))
        walk_avail = (walk_skim > 0) + np.diag(np.ones(nzones))

        base_bike_util = base_bike_util - 999 * (1 - bike_avail)
        base_walk_util = base_walk_util - 999 * (1 - walk_avail)

        ####################################
        # FIX: don't hard code these indices!
        #
        # use trip mode list
        ####################################
        motorized_trips = np.sum(base_trips[:, :, :5], 2)
        nonmotor_trips = np.sum(base_trips[:, :, 5:], 2)
        walk_trips = base_trips[:, :, 5]
        bike_trips = base_trips[:, :, 6]
        total_trips = motorized_trips + nonmotor_trips

        print('')
        print('segment ' + segment)
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

        ####################################
        # FIX: don't hard code these indices!
        #
        # use trip mode list
        ####################################
        build_trips = base_trips.copy()
        for motorized_idx in range(5):
            build_trips[:, :, motorized_idx] = \
                base_trips[:, :, motorized_idx] * \
                    np.nan_to_num(build_motor_trips / motorized_trips)
        build_trips[:, :, 5] = build_walk_trips
        build_trips[:, :, 6] = build_bike_trips

        # output.write_matrix_to_sqlite(build_trips,resources.application_config.base_sqlite_file,resources.mode_choice_config.trip_tables[idx],resources.mode_choice_config.modes)

        print('final trips')
        print('total motorized walk bike')
        print(int(np.sum(build_trips)),
              int(np.sum(build_motor_trips)),
              int(np.sum(build_walk_trips)),
              int(np.sum(build_bike_trips)))


if __name__ == '__main__':
    initial_demand()
