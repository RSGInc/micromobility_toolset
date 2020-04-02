import sqlite3

import numpy as np
import pandas as pd


class Skim():

    def __init__(self, data, mapping=None,
                 orig_col=None, dest_col=None,
                 col_names=None):
        """
        data: numpy array or pandas DataFrame
        mapping: mapping of matrix indices to id numbers
        """

        if not len(data.shape) in [2, 3]:
            raise IndexError(
                'input matrix must be 2 or 3 dimensions, not %s' % len(data.shape))

        if not data.shape[0] == data.shape[1]:
            raise IndexError('matrix dimensions 1 and 2 do not match: %s' % str(data.shape))

        self._matrix = data
        self._length = data.shape[0]

        self.set_mapping(mapping)
        self._set_num_cols()
        self.set_col_names(col_names)

    def set_mapping(self, mapping):

        # TODO: make sure map is all ints
        if not mapping:
            self.mapping = np.arange(self._length)

        elif isinstance(mapping, list):
            if not len(mapping) == self._length:
                raise IndexError('mapping of %s items cannot be applied to matrix '
                                 'with shape %s' % (len(mapping), str(self._matrix.shape)))

            self.mapping = np.array(mapping)

        else:
            raise TypeError('int or list for now')

    def _set_num_cols(self):

        if len(self._matrix.shape) == 2:
            self._num_cols = 1
        else:
            self._num_cols = self._matrix.shape[2]

    def set_col_names(self, col_names):

        if not col_names:
            self.col_names = None

        elif isinstance(col_names, list):
            assert len(col_names) == self._num_cols
            self.col_names = col_names

        else:
            raise TypeError('None or list for now')

    def to_numpy(self):

        return self._matrix

    def to_dataframe(self):

        multi_index = [
            np.repeat(self.mapping, self._length),
            np.tile(self.mapping, self._length)]

        data = self._matrix.reshape(self._length ** 2, self._num_cols)

        df = pd.DataFrame(data,
                          index=multi_index,
                          columns=self.col_names)

        return df

    def to_omx(self, filename):
        # need to check to make sure matrix entries are floats
        # also need to close file with a finally
        pass

    def to_sqlite(self, filename):
        # TODO: close file with a finally
        pass

    @classmethod
    def from_sqlite(cls, sqlite_file, table_name, orig_col, dest_col, columns=None):
        # TODO: close file with a finally
        # return cls(np.random.randn(2, 2))

        # open database cursor
        db_connection = sqlite3.connect(sqlite_file)

        matrix_df = pd.read_sql('select * from ' + table_name,
                                database_connection,
                                index_col=[orig_col, dest_col],
                                columns=columns)

        db_connection.close()

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

    @classmethod
    def from_omx(cls):
        pass


if __name__ == '__main__':

    np.random.seed(123)
    matrix = np.random.randn(4, 4, 3)

    print(matrix.shape)
    skim = Skim(matrix, mapping=[2, 4, 6, 8])

    # print(skim.to_numpy())
    print(skim.to_dataframe())
    # skim = Skim.from_sqlite()
    print(skim.to_numpy())
