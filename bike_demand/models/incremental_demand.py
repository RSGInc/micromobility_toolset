import argparse
import numpy as np

from activitysim.core import inject
from activitysim.core.config import setting

from ..utils import network
from ..utils.io import load_trip_matrix


def incremental_demand():
    # initialize configuration data
    trips_settings = inject.get_injectable('trips_settings')

    # store number of zones
    nzones = inject.get_injectable('num_zones')

    bike_skim = inject.get_injectable('bike_skim')

    # fix build walk skims to zero, not needed for incremental model
    walk_skim = np.zeros((nzones, nzones))

    # don't report zero divide in np arrayes
    np.seterr(divide='ignore', invalid='ignore')

    print("\nperforming model calculations...")

    # loop over market segments
    for segment in trips_settings.get('segments'):

        # read base trip table into matrix
        base_trips = load_trip_matrix(segment)

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
        base_bike_util = bike_avail * base_bike_util - 999 * ( 1 - bike_avail )
        base_walk_util = walk_avail * base_walk_util - 999 * ( 1 - walk_avail )
        build_bike_util = bike_avail * build_bike_util - 999 * ( 1 - bike_avail )
        build_walk_util = walk_avail * build_walk_util - 999 * ( 1 - walk_avail )

        # split full trip matrix and sum up into motorized, nonmotorized, walk, bike, and total

        ####################################
        # FIX: don't hard code these indices!
        #
        # use trip mode list
        ####################################
        motorized_trips = np.sum(base_trips[:,:,:5],2)
        nonmotor_trips = np.sum(base_trips[:,:,5:],2)
        walk_trips = base_trips[:,:,5]
        bike_trips = base_trips[:,:,6]
        total_trips = motorized_trips + nonmotor_trips

        # log base trips to console
        print('')
        print(('segment ' + segment))
        print('base trips')
        print('total motorized walk bike')
        print(int(np.sum(total_trips)), int(np.sum(motorized_trips)), int(np.sum(walk_trips)), int(np.sum(bike_trips)))

        # calculate logit denominator
        denom = (motorized_trips + walk_trips * np.exp(build_walk_util - base_walk_util) + bike_trips * np.exp( build_bike_util - base_bike_util ) )

        # perform incremental logit
        build_motor_trips = total_trips * np.nan_to_num( motorized_trips / denom )
        build_walk_trips = total_trips * np.nan_to_num( walk_trips * np.exp( build_walk_util - base_walk_util ) / denom )
        build_bike_trips = total_trips * np.nan_to_num( bike_trips * np.exp( build_bike_util - base_bike_util ) / denom )

        # combine into one trip matrix and proportionally scale motorized sub-modes
        ####################################
        # FIX: don't hard code these indices!
        #
        # use trip mode list
        ####################################
        build_trips = base_trips.copy()
        for motorized_idx in range(5):
            build_trips[:,:,motorized_idx] = base_trips[:,:,motorized_idx] * np.nan_to_num(build_motor_trips / motorized_trips)
        build_trips[:,:,5] = build_walk_trips
        build_trips[:,:,6] = build_bike_trips

        # write matrix to database
        # output.write_matrix_to_sqlite(build_trips,
        #                               build_sqlite_file,
        #                               table,
        #                               trips_settings.get('modes'))

        # log build trips to console
        print('build trips')
        print('total motorized walk bike')
        print(int(np.sum(build_trips)), int(np.sum(build_motor_trips)), int(np.sum(build_walk_trips)), int(np.sum(build_bike_trips)))


if __name__ == '__main__':
    incremental_demand()
