import argparse
import numpy as np

from activitysim.core.inject import get_injectable
from activitysim.core.config import setting

from ..utils import network
from ..utils.io import load_taz_matrix, save_taz_matrix


def incremental_demand():
    # initialize configuration data
    trips_settings = get_injectable('trips_settings')

    # store number of zones
    nzones = get_injectable('num_zones')

    bike_skim = get_injectable('bike_skim')

    # fix build walk skims to zero, not needed for incremental model
    walk_skim = np.zeros((nzones, nzones))

    # don't report zero divide in np arrayes
    np.seterr(divide='ignore', invalid='ignore')

    print("\nperforming model calculations...")

    # loop over market segments
    for segment in trips_settings.get('segments'):

        # use trips from previous step, if present
        base_trips = load_taz_matrix(segment)

        # calculate base walk and bike utilities
        base_bike_util = bike_skim * trips_settings.get('bike_skim_coef')
        base_walk_util = walk_skim * trips_settings.get('walk_skim_coef')

        # create initial build utilities
        build_bike_util = base_bike_util.copy()
        build_walk_util = base_walk_util.copy()

        # if not nhb, average PA and AP bike utilities
        if segment != 'nhb':
            base_bike_util = 0.5 * (base_bike_util + np.transpose(base_bike_util))
            build_bike_util = 0.5 * (build_bike_util + np.transpose(build_bike_util))

        # create 0-1 availability matrices when skim > 0
        walk_avail = (walk_skim > 0) + np.diag(np.ones(nzones))
        if segment != 'nhb':
            bike_avail = (bike_skim > 0) * np.transpose(bike_skim > 0) + np.diag(np.ones(nzones))
        else:
            bike_avail = (bike_skim > 0) + np.diag(np.ones(nzones))

        # non-available gets extreme negative utility
        base_bike_util = bike_avail * base_bike_util - 999 * (1 - bike_avail)
        base_walk_util = walk_avail * base_walk_util - 999 * (1 - walk_avail)
        build_bike_util = bike_avail * build_bike_util - 999 * (1 - bike_avail)
        build_walk_util = walk_avail * build_walk_util - 999 * (1 - walk_avail)

        # split full trip matrix and sum up into motorized, nonmotorized, walk, bike, and total
        midxs = get_injectable('auto_mode_indices')
        widxs = get_injectable('walk_mode_indices')
        bidxs = get_injectable('bike_mode_indices')

        motorized_trips = np.sum(np.take(base_trips, midxs, axis=2), 2)
        bike_trips = np.sum(np.take(base_trips, bidxs, axis=2), 2)
        walk_trips = np.sum(np.take(base_trips, widxs, axis=2), 2)
        total_trips = motorized_trips + bike_trips + walk_trips

        # log base trips to console
        print('')
        print(('segment ' + segment))
        print('base trips')
        print('total motorized walk bike')
        print(int(np.sum(total_trips)),
              int(np.sum(motorized_trips)),
              int(np.sum(walk_trips)),
              int(np.sum(bike_trips)))

        # calculate logit denominator
        denom = (motorized_trips + walk_trips *
                 np.exp(build_walk_util - base_walk_util) +
                 bike_trips * np.exp(build_bike_util - base_bike_util))

        # perform incremental logit
        build_motor_trips = total_trips * \
            np.nan_to_num(motorized_trips / denom)

        build_walk_trips = total_trips * \
            np.nan_to_num(walk_trips * np.exp(build_walk_util - base_walk_util) / denom)

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

        save_taz_matrix(build_trips, segment)

        # log build trips to console
        print('build trips')
        print('total motorized walk bike')
        print(int(np.sum(build_trips)),
              int(np.sum(build_motor_trips)),
              int(np.sum(build_walk_trips)),
              int(np.sum(build_bike_trips)))


if __name__ == '__main__':
    incremental_demand()
