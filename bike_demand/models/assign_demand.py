import csv

import numpy as np

from activitysim.core.inject import get_injectable
from activitysim.core.config import setting, output_file_path

from ..utils import network
from ..utils.io import load_trip_matrix


def assign_demand():

    # initialize configuration data
    trips_settings = get_injectable('trips_settings')

    nzones = get_injectable('num_zones')

    total_demand = np.zeros((nzones, nzones))

    print('getting demand matrices...')
    for segment in trips_settings.get('segments'):

        base_trips = load_trip_matrix(segment)

        ####################################
        # FIX: don't hard code these indices!
        #
        # use trip mode list
        ####################################
        bike_trips = base_trips[:, :, 6]

        if segment != 'nhb':
            bike_trips = 0.5 * (bike_trips + np.transpose(bike_trips))

        print('')
        print(('segment ' + segment))
        print('non-intrazonal bike trips')
        print(int(np.sum(bike_trips * (np.ones((nzones, nzones)) -
                                       np.diag(np.ones(nzones))))))

        total_demand = total_demand + bike_trips

    print('')
    print('assigning trips...')

    base_net = get_injectable('base_network')
    taz_nodes = get_injectable('taz_nodes')
    coef_bike = trips_settings.get('route_varcoef_bike')
    max_cost_bike = trips_settings.get('max_cost_bike')

    base_net.assign_trip_matrix(trips=total_demand,
                                load_name='bike_vol',
                                taz_nodes=taz_nodes,
                                varcoef=coef_bike,
                                max_cost=max_cost_bike)

    with open(output_file_path('bike_vol.csv'), 'w') as f:
        writer = csv.writer(f)

        print('writing results...')
        for a in base_net.adjacency:
            for b in base_net.adjacency[a]:
                writer.writerow([a, b, base_net.get_edge_attribute_value((a, b), 'bike_vol')])


if __name__ == '__main__':
    assign_demand()
