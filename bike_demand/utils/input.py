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
