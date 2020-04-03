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
