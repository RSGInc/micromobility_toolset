import csv

import numpy as np

from activitysim.core import inject
from activitysim.core.config import (
    setting,
    data_file_path,
    output_file_path,
    read_model_settings)

from ..utils import (network, output)
from ..utils.input import read_taz_from_sqlite, read_matrix_from_sqlite


def assign_demand():

    # initialize configuration data
    network_settings = read_model_settings('network.yaml')
    trips_settings = read_model_settings('trips.yaml')

    # store number of zones
    max_zone = setting('max_zone') + 1

    # read network data
    base_sqlite_file = data_file_path(setting('base_sqlite_file'))
    build_sqlite_file = data_file_path(setting('build_sqlite_file'))
    base_net = network.Network(network_settings, base_sqlite_file)
    build_net = network.Network(network_settings, build_sqlite_file)

    add_derived_network_attributes(base_net)

    taz_data = read_taz_from_sqlite(base_sqlite_file,
                                    setting('taz_table_name'),
                                    index_col=setting('taz_table_name'),
                                    columns=[setting('taz_node_column'),
                                             setting('taz_county_column')])

    taz_nodes = {}
    taz_county = {}
    for taz in taz_data:
        taz_nodes[taz] = taz_data[taz][setting('taz_node_column')]
        taz_county[taz] = taz_data[taz][setting('taz_county_column')]

    total_demand = np.zeros((max_zone, max_zone))

    print('getting demand matrices...')
    for table in trips_settings.get('trip_tables'):

        base_trips = read_matrix_from_sqlite(
            base_sqlite_file, table,
            trips_settings.get('trip_ataz_col'), trips_settings.get('trip_ptaz_col'))

        if base_trips.size == 0:
            print('%s is empty or missing' % table)
            continue

        bike_trips = base_trips[:, :, 6]

        if table != 'nhbtrip':
            bike_trips = 0.5 * (bike_trips + np.transpose(bike_trips))

        print('')
        print(('segment ' + table))
        print('non-intrazonal bike trips')
        print(int(np.sum(bike_trips * (np.ones((max_zone, max_zone)) -
                                       np.diag(np.ones(max_zone))))))

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


def add_derived_network_attributes(net):

    net.add_edge_attribute('d0')
    net.add_edge_attribute('d1')
    net.add_edge_attribute('d2')
    net.add_edge_attribute('d3')
    net.add_edge_attribute('dne1')
    net.add_edge_attribute('dne2')
    net.add_edge_attribute('dne3')
    net.add_edge_attribute('dw')
    net.add_edge_attribute('riseft')
    net.add_edge_attribute('auto_permit')
    net.add_edge_attribute('bike_exclude')
    net.add_edge_attribute('dloc')
    net.add_edge_attribute('dcol')
    net.add_edge_attribute('dart')
    net.add_edge_attribute('dne3loc')
    net.add_edge_attribute('dne2art')
    net.add_edge_attribute('bike_vol')

    for a in net.adjacency:
        for b in net.adjacency[a]:
            distance = net.get_edge_attribute_value((a,b),'distance')
            bike_class = net.get_edge_attribute_value((a,b),'bike_class')
            lanes = net.get_edge_attribute_value((a,b),'lanes')
            # from_elev = net.get_edge_attribute_value((a,b),'from_elev')
            # to_elev = net.get_edge_attribute_value((a,b),'to_elev')
            link_type = net.get_edge_attribute_value((a,b),'link_type')
            fhwa_fc = net.get_edge_attribute_value((a,b),'fhwa_fc')
            net.set_edge_attribute_value( (a,b), 'd0', distance * ( bike_class == 0 and lanes > 0 ) )
            net.set_edge_attribute_value( (a,b), 'd1', distance * ( bike_class == 1 ) )
            net.set_edge_attribute_value( (a,b), 'd2', distance * ( bike_class == 2 ) )
            net.set_edge_attribute_value( (a,b), 'd3', distance * ( bike_class == 3 ) )
            net.set_edge_attribute_value( (a,b), 'dne1', distance * ( bike_class != 1 ) )
            net.set_edge_attribute_value( (a,b), 'dne2', distance * ( bike_class != 2 ) )
            net.set_edge_attribute_value( (a,b), 'dne3', distance * ( bike_class != 3 ) )
            net.set_edge_attribute_value( (a,b), 'dw', distance * ( bike_class == 0 and lanes == 0 ) )
            # net.set_edge_attribute_value( (a,b), 'riseft',  max(to_elev - from_elev,0) )
            net.set_edge_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
            net.set_edge_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
            net.set_edge_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
            net.set_edge_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,17] ) )
            net.set_edge_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,16,77] ) )
            net.set_edge_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
            net.set_edge_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,16,77] ) * ( bike_class != 2 ) )
            net.set_edge_attribute_value( (a,b), 'bike_vol',0)

    net.add_dual_attribute('thru_centroid')
    net.add_dual_attribute('l_turn')
    net.add_dual_attribute('u_turn')
    net.add_dual_attribute('r_turn')
    net.add_dual_attribute('turn')
    net.add_dual_attribute('thru_intersec')
    net.add_dual_attribute('thru_junction')

    net.add_dual_attribute('path_onoff')

    for edge1 in net.dual:
        for edge2 in net.dual[edge1]:

            traversal_type = net.traversal_type(edge1,edge2,'auto_permit')

            net.set_dual_attribute_value(edge1,edge2,'thru_centroid', 1 * (traversal_type == 0) )
            net.set_dual_attribute_value(edge1,edge2,'u_turn', 1 * (traversal_type == 3 ) )
            net.set_dual_attribute_value(edge1,edge2,'l_turn', 1 * (traversal_type in [5,7,10,13]) )
            net.set_dual_attribute_value(edge1,edge2,'r_turn', 1 * (traversal_type in [4,6,9,11]) )
            net.set_dual_attribute_value(edge1,edge2,'turn', 1 * (traversal_type in [3,4,5,6,7,9,10,11,13]) )
            net.set_dual_attribute_value(edge1,edge2,'thru_intersec', 1 * (traversal_type in [8,12]) )
            net.set_dual_attribute_value(edge1,edge2,'thru_junction', 1 * (traversal_type == 14) )

            path1 = ( net.get_edge_attribute_value(edge1,'bike_class') == 1 )
            path2 = ( net.get_edge_attribute_value(edge2,'bike_class') == 1 )

            net.set_dual_attribute_value(edge1,edge2,'path_onoff', 1 * ( (path1 + path2) == 1 ) )


if __name__ == '__main__':
    assign_demand()
