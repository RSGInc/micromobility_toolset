import csv

import numpy as np

from activitysim.core import inject
from activitysim.core.config import setting, output_file_path

from ..utils import network
from ..utils.io import read_matrix


def assign_demand():

    # initialize configuration data
    network_settings = inject.get_injectable('network_settings')
    trips_settings = inject.get_injectable('trips_settings')

    taz_data = inject.get_injectable('taz_data')

    # store number of zones
    nzones = len(taz_data)

    # read network data
    base_net = network.Network(network_settings)
    build_net = network.Network(network_settings)

    add_derived_network_attributes(base_net)

    taz_nodes = {}
    taz_county = {}
    for taz in taz_data:
        taz_nodes[taz] = taz_data[taz][setting('taz_node_column')]
        taz_county[taz] = taz_data[taz][setting('taz_county_column')]

    total_demand = np.zeros((nzones, nzones))

    print('getting demand matrices...')
    for segment in trips_settings.get('segments'):

        table = segment + trips_settings.get('trip_table_suffix')

        base_trips = read_matrix(table)

        bike_trips = base_trips[:, :, 6]

        if table != 'nhbtrip':
            bike_trips = 0.5 * (bike_trips + np.transpose(bike_trips))

        print('')
        print(('segment ' + table))
        print('non-intrazonal bike trips')
        print(int(np.sum(bike_trips * (np.ones((nzones, nzones)) -
                                       np.diag(np.ones(nzones))))))

        total_demand = total_demand + bike_trips

    print('')
    print('assigning trips...')
    base_net.load_trip_matrix(total_demand, 'bike_vol', taz_nodes,
                              trips_settings.get('route_varcoef_bike'),
                              trips_settings.get('max_cost_bike'))

    with open(output_file_path('bike_vol.csv'), 'w') as f:
        writer = csv.writer(f)

        print('writing results...')
        for a in base_net.adjacency:
            for b in base_net.adjacency[a]:
                writer.writerow([a, b, base_net.get_edge_attribute_value((a, b), 'bike_vol')])


if __name__ == '__main__':
    assign_demand()
