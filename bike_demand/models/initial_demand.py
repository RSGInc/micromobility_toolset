import numpy as np

from activitysim.core import inject
from activitysim.core.config import (
    setting,
    data_file_path,
    output_file_path,
    read_model_settings)

from ..utils import network
from ..utils.input import read_matrix


def initial_demand():
    # initialize configuration data
    network_settings = read_model_settings('network.yaml')
    trips_settings = read_model_settings('trips.yaml')

    # read network data
    base_sqlite_file = data_file_path(setting('base_sqlite_file'))
    build_sqlite_file = data_file_path(setting('build_sqlite_file'))
    base_net = network.Network(network_settings, base_sqlite_file)
    build_net = network.Network(network_settings, build_sqlite_file)

    add_derived_network_attributes(base_net)

    taz_data = inject.get_injectable('taz_data')

    nzones = len(taz_data)

    taz_nodes = {}
    taz_county = {}
    for taz in taz_data:
        taz_nodes[taz] = taz_data[taz][setting('taz_node_column')]
        taz_county[taz] = taz_data[taz][setting('taz_county_column')]

    if setting('read_base_skims_from_disk'):
        print('reading skims from disk...')
        # base_walk_skim = read_matrix_from_sqlite(resources,'walk_skim',base_sqlite_file)
        # base_bike_skim = read_matrix_from_sqlite(resources,'bike_skim',.base_sqlite_file)
    else:
        print('skimming network...')
        base_walk_skim = base_net.get_skim_matrix(
            taz_nodes,
            trips_settings.get('route_varcoef_walk'),
            max_cost=trips_settings.get('max_cost_walk'))

        base_walk_skim = base_walk_skim * (np.ones((nzones, nzones)) -
                                           np.diag(np.ones(nzones)))

        base_bike_skim = base_net.get_skim_matrix(
            taz_nodes,
            trips_settings.get('route_varcoef_bike'),
            max_cost=trips_settings.get('max_cost_bike'))

        base_bike_skim = base_bike_skim * (np.ones((nzones, nzones)) -
                                           np.diag(np.ones(nzones)))

        print('writing results...')
        # output.write_matrix_to_sqlite(base_walk_skim,resources.application_config.base_sqlite_file,'walk_skim',['value'])
        # output.write_matrix_to_sqlite(base_bike_skim,resources.application_config.base_sqlite_file,'bike_skim',['value'])

    np.seterr(divide='ignore', invalid='ignore')

    print('performing model calculations...')
    for segment in trips_settings.get('segments'):

        trip_table = segment + trips_settings.get('trip_table_suffix')
        motutil_table = segment + trips_settings.get('motorized_util_table_suffix')

        # read in trip tables
        base_trips = read_matrix(trip_table)

        if base_trips.size == 0:
            print('\n%s is empty or missing' % trip_table)
            continue

        base_motor_util = read_matrix(motutil_table)

        if base_motor_util.size == 0:
            print('\n%s is empty or missing' % motutil_table)
            continue

        base_bike_util = base_bike_skim * trips_settings.get('bike_skim_coef')
        base_walk_util = base_walk_skim * trips_settings.get('walk_skim_coef')

        if segment != 'nhb':
            base_bike_util = 0.5 * (base_bike_util + np.transpose(base_bike_util))

        base_motor_util = base_motor_util * (np.ones((nzones, nzones)) -
                                             np.diag(np.ones(nzones)))

        base_bike_util = base_bike_util + trips_settings.get('bike_asc')[segment]
        base_walk_util = base_walk_util + trips_settings.get('walk_asc')[segment]
        base_bike_util = base_bike_util + trips_settings.get('bike_intrazonal')
        base_walk_util = base_walk_util + trips_settings.get('walk_intrazonal')

        bike_avail = (base_bike_skim > 0) + np.diag(np.ones(nzones))
        walk_avail = (base_walk_skim > 0) + np.diag(np.ones(nzones))

        base_bike_util = base_bike_util - 999 * (1 - bike_avail)
        base_walk_util = base_walk_util - 999 * (1 - walk_avail)

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


def add_derived_network_attributes(net):

    net.add_edge_attribute('d0')
    net.add_edge_attribute('d1')
    net.add_edge_attribute('d2')
    net.add_edge_attribute('d3')
    net.add_edge_attribute('dne1')
    net.add_edge_attribute('dne2')
    net.add_edge_attribute('dne3')
    net.add_edge_attribute('dw')
    #net.add_edge_attribute('riseft')
    net.add_edge_attribute('auto_permit')
    net.add_edge_attribute('bike_exclude')
    net.add_edge_attribute('dloc')
    net.add_edge_attribute('dcol')
    net.add_edge_attribute('dart')
    net.add_edge_attribute('dne3loc')
    net.add_edge_attribute('dne2art')

    for a in net.adjacency:
        for b in net.adjacency[a]:
            distance = net.get_edge_attribute_value((a,b),'distance')
            bike_class = net.get_edge_attribute_value((a,b),'bike_class')
            lanes = net.get_edge_attribute_value((a,b),'lanes')
            #from_elev = net.get_edge_attribute_value((a,b),'from_elev')
            #to_elev = net.get_edge_attribute_value((a,b),'to_elev')
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
            #net.set_edge_attribute_value( (a,b), 'riseft',  max(to_elev - from_elev,0) )
            net.set_edge_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
            net.set_edge_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
            net.set_edge_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
            net.set_edge_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,16,17] ) )
            net.set_edge_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) )
            net.set_edge_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
            net.set_edge_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) * ( bike_class != 2 ) )

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
    initial_demand()
