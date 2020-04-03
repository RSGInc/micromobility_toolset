import argparse
import numpy as np

from activitysim.core import inject
from activitysim.core.config import setting

from ..utils import network
from ..utils.io import read_matrix


def incremental_demand():
    # initialize configuration data
    network_settings = inject.get_injectable('network_settings')
    trips_settings = inject.get_injectable('trips_settings')

    # store number of zones
    taz_data = inject.get_injectable('taz_data')
    nzones = len(taz_data)

    # read network data
    base_net = network.Network(network_settings)
    build_net = network.Network(network_settings)

    # calculate derived network attributes
    coef_walk = trips_settings.get('route_varcoef_walk')
    coef_bike = trips_settings.get('route_varcoef_bike')
    add_derived_network_attributes(base_net, coef_walk, coef_bike)
    add_derived_network_attributes(build_net, coef_walk, coef_bike)

    taz_nodes = {}
    taz_county = {}
    for taz in taz_data:
        taz_nodes[taz] = taz_data[taz][setting('taz_node_column')]
        taz_county[taz] = taz_data[taz][setting('taz_county_column')]


    print('skimming base network...')
    base_bike_skim = base_net.get_skim_matrix(taz_nodes,
                                              trips_settings.get('route_varcoef_bike'),
                                              trips_settings.get('max_cost_bike'))
    base_bike_skim = base_bike_skim * \
                        (np.ones((nzones, nzones)) - np.diag(np.ones(nzones)))

    print('writing results...')
    # output.write_matrix_to_sqlite(base_bike_skim,
    #                               base_sqlite_file,
    #                               'bike_skim',
    #                               ['value'])


    print('skimming build network...')
    build_bike_skim = build_net.get_skim_matrix(taz_nodes,
                                                trips_settings.get('route_varcoef_bike'),
                                                trips_settings.get('max_cost_bike'))
    build_bike_skim = build_bike_skim * \
                        (np.ones((nzones, nzones)) - np.diag(np.ones(nzones)))

    # print('writing results...')
    # output.write_matrix_to_sqlite(build_bike_skim,
    #                               build_sqlite_file,
    #                               'bike_skim',
    #                               ['value'])

    # fix build walk skims to zero, not needed for incremental model
    base_walk_skim = np.zeros((nzones, nzones))
    build_walk_skim = np.zeros((nzones, nzones))

    # don't report zero divide in np arrayes
    np.seterr(divide='ignore', invalid='ignore')

    print("\nperforming model calculations...")

    # loop over market segments
    for segment in trips_settings.get('segments'):

        table = segment + trips_settings.get('trip_table_suffix')

        # read base trip table into matrix
        base_trips = read_matrix(table)

        # calculate base walk and bike utilities
        base_bike_util = base_bike_skim * trips_settings.get('bike_skim_coef')
        base_walk_util = base_walk_skim * trips_settings.get('walk_skim_coef')

        # calculate build walk and bike utilities
        build_bike_util = build_bike_skim * trips_settings.get('bike_skim_coef')
        build_walk_util = build_walk_skim * trips_settings.get('walk_skim_coef')

        # if not nhb, average PA and AP bike utilities
        if table != 'nhbtrip':
            base_bike_util = 0.5 * (base_bike_util + np.transpose(base_bike_util))
            build_bike_util = 0.5 * (build_bike_util + np.transpose(build_bike_util))

        # create 0-1 availability matrices when skim > 0
        walk_avail = (base_walk_skim > 0) + np.diag(np.ones(nzones))
        if table != 'nhbtrip':
            bike_avail = (base_bike_skim > 0) * np.transpose(base_bike_skim > 0) + np.diag(np.ones(nzones))
        else:
            bike_avail = (base_bike_skim > 0) + np.diag(np.ones(nzones))

        # non-available gets extreme negative utility
        base_bike_util = bike_avail * base_bike_util - 999 * ( 1 - bike_avail )
        base_walk_util = walk_avail * base_walk_util - 999 * ( 1 - walk_avail )
        build_bike_util = bike_avail * build_bike_util - 999 * ( 1 - bike_avail )
        build_walk_util = walk_avail * build_walk_util - 999 * ( 1 - walk_avail )

        # split full trip matrix and sum up into motorized, nonmotorized, walk, bike, and total
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

        # # perform tracing if desired
        # if resources.application_config.trace == True and resources.application_config.trace_segment == resources.mode_choice_config.trip_tables[idx]:
        #
        #     ptaz = resources.application_config.trace_ptaz
        #     ataz = resources.application_config.trace_ataz
        #
        #     print('')
        #     print('TRACE')
        #     print('ptaz: ', ptaz)
        #     print('ataz: ', ataz)
        #
        #     print('base pa')
        #     path = base_net.single_source_dijkstra(taz_nodes[ptaz],resources.mode_choice_config.route_varcoef_bike,target=taz_nodes[ataz])[1][taz_nodes[ataz]]
        #     print('path: ', path)
        #     for var in resources.mode_choice_config.route_varcoef_bike:
        #         print(var, base_net.path_trace(path,var))
        #
        #     print('')
        #     print('build pa')
        #     path = build_net.single_source_dijkstra(taz_nodes[ptaz],resources.mode_choice_config.route_varcoef_bike,target=taz_nodes[ataz])[1][taz_nodes[ataz]]
        #     print('path: ', path)
        #     for var in resources.mode_choice_config.route_varcoef_bike:
        #         print(var, build_net.path_trace(path,var))
        #
        #     print('')
        #     print('base ap')
        #     path = base_net.single_source_dijkstra(taz_nodes[ataz],resources.mode_choice_config.route_varcoef_bike,target=taz_nodes[ptaz])[1][taz_nodes[ptaz]]
        #     print('path: ', path)
        #     for var in resources.mode_choice_config.route_varcoef_bike:
        #         print(var, base_net.path_trace(path,var))
        #
        #     print('')
        #     print('build ap')
        #     path = build_net.single_source_dijkstra(taz_nodes[ataz],resources.mode_choice_config.route_varcoef_bike,target=taz_nodes[ptaz])[1][taz_nodes[ptaz]]
        #     print('path: ', path)
        #     for var in resources.mode_choice_config.route_varcoef_bike:
        #         print(var, build_net.path_trace(path,var))
        #
        #     print('')
        #     print('chg. bike util')
        #     print(build_bike_util[ataz-1][ptaz-1] - base_bike_util[ataz-1][ptaz-1])
        #
        #     print('')
        #     print('base trips')
        #     print('da s2 s3 wt dt wk bk')
        #     print(base_trips[ptaz-1,ataz-1,0], base_trips[ptaz-1,ataz-1,1], base_trips[ptaz-1,ataz-1,2], base_trips[ptaz-1,ataz-1,3], base_trips[ptaz-1,ataz-1,4], base_trips[ptaz-1,ataz-1,5], base_trips[ptaz-1,ataz-1,6])
        #
        #     print('')
        #     print('build trips')
        #     print('da s2 s3 wt dt wk bk')
        #     print(build_trips[ptaz-1,ataz-1,0], build_trips[ptaz-1,ataz-1,1], build_trips[ptaz-1,ataz-1,2], build_trips[ptaz-1,ataz-1,3], build_trips[ptaz-1,ataz-1,4], build_trips[ptaz-1,ataz-1,5], build_trips[ptaz-1,ataz-1,6])


def add_derived_network_attributes(net, coef_walk, coef_bike):
    """add network attributes that are combinations of attributes from sqlite database"""

    # add new link attribute columns
    net.add_edge_attribute('d0') # distance on ordinary streets, miles
    net.add_edge_attribute('d1') # distance on bike paths
    net.add_edge_attribute('d2') # distance on bike lanes
    net.add_edge_attribute('d3') # distance on bike routes
    net.add_edge_attribute('dne1') # distance not on bike paths
    net.add_edge_attribute('dne2') # distance not on bike lanes
    net.add_edge_attribute('dne3') # distance not on bike routes
    net.add_edge_attribute('dw') # distance wrong way
    #gain now comes directly from sqlite
    #net.add_edge_attribute('riseft')
    net.add_edge_attribute('auto_permit') # autos permitted
    net.add_edge_attribute('bike_exclude') # bikes excluded
    net.add_edge_attribute('dloc') # distance on local streets
    net.add_edge_attribute('dcol') # distance on collectors
    net.add_edge_attribute('dart') # distance on arterials
    net.add_edge_attribute('dne3loc') # distance on locals with no bike route
    net.add_edge_attribute('dne2art') # distance on arterials with no bike lane

    # loop over edges and calculate derived values
    for a in net.adjacency:
        for b in net.adjacency[a]:
            distance = net.get_edge_attribute_value((a,b),'distance')
            bike_class = net.get_edge_attribute_value((a,b),'bike_class')
            lanes = net.get_edge_attribute_value((a,b),'lanes')
            #gain now comes directly from sqlite
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
            #gain now comes directly from sqlite
            #net.set_edge_attribute_value( (a,b), 'riseft',  max(to_elev - from_elev,0) )
            net.set_edge_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
            net.set_edge_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
            net.set_edge_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
            net.set_edge_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,16,17] ) )
            net.set_edge_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) )
            net.set_edge_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
            net.set_edge_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) * ( bike_class != 2 ) )

    # add new dual (link-to-link) attribute columns
    net.add_dual_attribute('thru_centroid') # from centroid connector to centroid connector
    net.add_dual_attribute('l_turn') # left turn
    net.add_dual_attribute('u_turn') # u turn
    net.add_dual_attribute('r_turn') # right turn
    net.add_dual_attribute('turn') # turn
    net.add_dual_attribute('thru_intersec') # through a highway intersection
    net.add_dual_attribute('thru_junction') # through a junction

    net.add_dual_attribute('path_onoff') # movement in between bike path and other type

    net.add_dual_attribute('walk_cost') # total walk generalized cost
    net.add_dual_attribute('bike_cost') # total bike generalized cost

    # loop over pairs of edges and set attribute values
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

            net.set_dual_attribute_value(edge1,edge2,'walk_cost',net.calculate_variable_cost(edge1,edge2,coef_walk,0.0) )
            net.set_dual_attribute_value(edge1,edge2,'bike_cost',net.calculate_variable_cost(edge1,edge2,coef_bike,0.0) )


if __name__ == '__main__':
    incremental_demand()
