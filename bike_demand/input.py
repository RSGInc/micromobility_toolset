import csv
import sqlite3
import time

import numpy as np

from . import (choice_set, network, config, output)

def read_taz_from_sqlite(config):

    result = {}

    # open database cursor
    database_connection = sqlite3.connect(config.application_config.base_sqlite_file)
    database_connection.row_factory  = sqlite3.Row
    database_cursor = database_connection.cursor()

    # execute select of link table
    database_cursor.execute('select * from ' + config.application_config.taz_table_name)

    # loop over database records
    while True:

        # get next record
        row = database_cursor.fetchone()

        if row is None:
            # if no more records we're done
            break
        else:
            taz = row[list(row.keys()).index(config.application_config.taz_taz_column)]
            node = row[list(row.keys()).index(config.application_config.taz_node_column)]
            county = row[list(row.keys()).index(config.application_config.taz_county_column)]
            result[taz] =  {'node': node, 'county': county}

    return result


def read_matrix_from_sqlite(config,table_name,sqlite_file):

    # open database cursor
    database_connection = sqlite3.connect(sqlite_file)
    database_cursor = database_connection.cursor()

    # execute select of link table
    database_cursor.execute('select * from ' + table_name)

    rows = database_cursor.fetchall()
    if len(rows) == 0:
        return np.array([])

    if len(rows[0])>3:
        dim = (config.application_config.num_zones+1,config.application_config.num_zones+1,len(rows[0])-2)
    else:
        dim = (config.application_config.num_zones+1,config.application_config.num_zones+1)

    trip_matrix = np.zeros(dim)

    if len(rows[0])>3:
        for row in rows:
            trip_matrix[row[0],row[1],:] = row[2:]
    else:
        for row in rows:
            trip_matrix[row[0],row[1]] = row[2]

    return trip_matrix


def get_skim_matrix(net, taz_nodes, varcoef, max_cost=None):
    """skim network net starting from taz nodes in taz_nodes, with variable coefficients varcoef
    until max_cost is reached, return matrix
    """

    max_taz = max(taz_nodes.keys())
    skim_matrix = np.zeros((max_taz+1, max_taz+1))

    for i in taz_nodes.keys():

        centroid = taz_nodes[i]
        costs = net.single_source_dijkstra(centroid, varcoef, max_cost=max_cost)[0]

        for j in taz_nodes.keys():

            if taz_nodes[j] in costs:
                skim_matrix[i, j] = costs[taz_nodes[j]]

    return skim_matrix


def load_trip_matrix(net,trips,load_name,taz_nodes,varcoef,max_cost=None):

    max_taz = max( taz_nodes.keys() )

    for i in taz_nodes.keys():

        centroid = taz_nodes[i]
        paths = net.single_source_dijkstra(centroid,varcoef,max_cost=max_cost)[1]

        for j in taz_nodes.keys():

            target = taz_nodes[j]

            if target in paths and trips[i,j] > 0:
                for k in range(len(paths[target])-1):
                    edge = (paths[target][k],paths[target][k+1])
                    net.set_edge_attribute_value(edge,load_name,trips[i,j]+net.get_edge_attribute_value(edge,load_name))
