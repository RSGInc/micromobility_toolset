import sqlite3

import numpy as np
import pandas as pd


def read_taz_from_sqlite(sqlite_file, table_name, index_col=None, columns=None):

    # open database cursor
    database_connection = sqlite3.connect(sqlite_file)

    taz_df = pd.read_sql('select * from ' + table_name,
                         database_connection,
                         index_col=index_col,
                         columns=columns)

    database_connection.close()

    return taz_df.T.to_dict()


def read_matrix_from_sqlite(sqlite_file, table_name, orig_col, dest_col, columns=None):

    # open database cursor
    database_connection = sqlite3.connect(sqlite_file)

    matrix_df = pd.read_sql('select * from ' + table_name,
                            database_connection,
                            index_col=[orig_col, dest_col],
                            columns=columns)

    database_connection.close()

    atazs = matrix_df.index.get_level_values(orig_col)
    ptazs = matrix_df.index.get_level_values(dest_col)

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
    else:
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
