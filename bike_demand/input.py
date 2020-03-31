import csv
import sqlite3
import time

import numpy as np
import pandas as pd

from . import (choice_set, network, config, output)

def read_taz_from_sqlite(config):

    # open database cursor
    database_connection = sqlite3.connect(config.application_config.base_sqlite_file)

    taz_df = pd.read_sql('select * from ' + config.application_config.taz_table_name,
                         database_connection,
                         index_col=config.application_config.taz_taz_column,
                         columns=[config.application_config.taz_node_column,
                                  config.application_config.taz_county_column])

    database_connection.close()

    return taz_df.T.to_dict()


def read_matrix_from_sqlite(config, table_name, sqlite_file):

    # open database cursor
    database_connection = sqlite3.connect(sqlite_file)

    matrix_df = pd.read_sql('select * from ' + table_name,
                            database_connection,
                            index_col=['ataz', 'ptaz'])

    database_connection.close()

    atazs = matrix_df.index.get_level_values('ataz')
    ptazs = matrix_df.index.get_level_values('ptaz')

    if matrix_df.shape[0] == 0:
        return np.array([])

    matrix_dim = max(list(atazs) + list(ptazs)) + 1

    if matrix_df.shape[1] > 1:
        dim = (matrix_dim, matrix_dim, matrix_df.shape[1])
    else:
        dim = (matrix_dim, matrix_dim)

    trip_matrix = np.zeros(dim)

    if matrix_df.shape[1] > 1:
        trip_matrix[atazs, ptazs, :] = matrix_df.iloc[:, 0:].to_numpy()
        # for row in rows:
        #     trip_matrix[row[0],row[1],:] = row[2:]
    else:
        atazs = matrix_df.index.get_level_values(0)
        ptazs = matrix_df.index.get_level_values(1)
        trip_matrix[atazs, ptazs] = matrix_df.iloc[:, 0].to_numpy()

    return trip_matrix


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
