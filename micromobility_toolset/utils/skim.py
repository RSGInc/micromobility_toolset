import sqlite3

import numpy as np
import pandas as pd


class Skim():

    def __init__(self, data, **kwargs):
        """
        data: numpy array or pandas DataFrame
        """

        if isinstance(data, np.ndarray):
            self.from_numpy(data, **kwargs)

        elif isinstance(data, pd.DataFrame):
            self.from_dataframe(data, **kwargs)

        else:
            raise TypeError('data must be a numpy array or pandas DataFrame')

    def from_numpy(self, data, mapping=None,
                   orig_col=None, dest_col=None,
                   col_names=None):
        """
        data: 2- or 3- dimensional numpy array
        mapping: listlike of matrix index ids (int)
        orig_col: str, name of 1st dimension
        dest_col: str, name of 2nd dimension
        col_names: list of str, names for higher dimensions
        """

        if not len(data.shape) in [2, 3]:
            raise IndexError(f'input matrix must be 2 or 3 dimensions, not {len(data.shape)}')

        if not data.shape[0] == data.shape[1]:
            raise IndexError(f'matrix dimensions 1 and 2 do not match: {data.shape}')

        self._matrix = data
        self._length = data.shape[0]

        self._set_mapping(mapping)
        self._set_num_cols()
        self._set_index(orig_col, dest_col)
        self._set_col_names(col_names)

    def from_dataframe(self, data, mapping=None,
                       orig_col=None, dest_col=None,
                       col_names=None):

        if isinstance(data.index, pd.MultiIndex):
            # FIX: what if the index names already exist?
            data.index.names = [orig_col, dest_col]

        else:
            # FIX: avoid expensive reindexing?
            data.set_index([orig_col, dest_col], inplace=True)

        o_vals = data.index.get_level_values(0)
        d_vals = data.index.get_level_values(1)

        index_vals = sorted(list(set(list(o_vals) + list(d_vals))))

        if mapping:
            # if not all(i in index_vals for i in mapping):
            #     raise IndexError('DataFrame index is incomplete for given mapping')

            # only retrieve rows from mapping
            o_vals = [val for val in o_vals if val in mapping]
            d_vals = [val for val in d_vals if val in mapping]

            data = data.reindex([o_vals, d_vals], copy=False, fill_value=0)

        else:
            mapping = index_vals

        # if matrix_df.shape[0] == 0:
        #     return np.array([])

        matrix_length = len(mapping)

        if col_names:
            data = data[col_names]

        else:
            col_names = list(data.columns)

        if data.shape[1] > 1:
            dim = (matrix_length, matrix_length, data.shape[1])
        else:
            dim = (matrix_length, matrix_length)

        # # Should we allow non-floats? They are much bulkier
        # if any(data.dtypes.isin([np.dtype('object')])):
        #     dtype = np.dtype('object')
        # else:
        #     dtype = np.dtype('float')

        np_matrix = np.zeros(dim)  # .astype(dtype)

        o_index = [mapping.index(i) for i in o_vals]
        d_index = [mapping.index(i) for i in d_vals]

        if data.shape[1] > 1:
            np_matrix[o_index, d_index, :] = data.iloc[:, 0:].to_numpy()
        else:
            np_matrix[o_index, d_index] = data.iloc[:, 0].to_numpy()

        self._matrix = np_matrix
        self._length = matrix_length

        self._set_mapping(mapping)
        self._set_num_cols()
        self._set_index(orig_col, dest_col)
        self._set_col_names(col_names)

    def _set_mapping(self, mapping):

        # TODO: make sure map is all ints
        if not mapping:
            self._mapping = np.arange(self._length)

        elif isinstance(mapping, list) or isinstance(mapping, np.ndarray):
            if not len(mapping) == self._length:
                raise IndexError(f'mapping of {len(mapping)} items cannot be applied to matrix '
                                 f'with shape {self._matrix.shape}')

            self._mapping = np.array(mapping)

        else:
            raise TypeError('int or list for now')

    def _set_num_cols(self):

        if len(self._matrix.shape) == 2:
            self._num_cols = 1
        else:
            self._num_cols = self._matrix.shape[2]

    def _set_index(self, orig_col, dest_col):

        self._orig_col = orig_col
        self._dest_col = dest_col

    def _set_col_names(self, col_names):

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

        multi_index = pd.MultiIndex.from_arrays(
            [
                np.repeat(self._mapping, self._length),
                np.tile(self._mapping, self._length)
            ],
            names=[self._orig_col, self._dest_col])

        data = self._matrix.reshape(self._length ** 2, self._num_cols)

        df = pd.DataFrame(data,
                          index=multi_index,
                          columns=self.col_names)

        return df

    def to_omx(self, filename):
        # need to check to make sure matrix entries are floats
        # also need to close file with a finally
        pass

    def to_sqlite(self, filename, table_name, **kwargs):

        db_connection = sqlite3.connect(filename)

        try:
            self.to_dataframe().to_sql(name=table_name,
                                       con=db_connection,
                                       **kwargs)

        finally:
            db_connection.close()

    def to_csv(self, filename, **kwargs):

        self.to_dataframe().to_csv(filename, **kwargs)

    @classmethod
    def from_omx(cls):
        pass

    @classmethod
    def from_sqlite(cls, sqlite_file, table_name,
                    orig_col, dest_col,
                    col_names=None,
                    mapping=None):

        # open database cursor
        db_connection = sqlite3.connect(sqlite_file)

        try:
            matrix_df = pd.read_sql(f'select * from {table_name}',
                                    db_connection,
                                    index_col=[orig_col, dest_col],
                                    columns=col_names)

            return cls(matrix_df,
                       orig_col=orig_col,
                       dest_col=dest_col,
                       col_names=col_names,
                       mapping=mapping)

        finally:
            db_connection.close()

    @classmethod
    def from_csv(cls, csv_file,
                 orig_col, dest_col,
                 col_names=None,
                 mapping=None):

        if col_names:
            # usecols doesn't include index_col values by default
            columns = list(col_names) + list([orig_col, dest_col])
        else:
            # None will include all columns
            columns = None

        matrix_df = pd.read_csv(csv_file,
                                index_col=[orig_col, dest_col],
                                usecols=None)

        return cls(matrix_df,
                   orig_col=orig_col,
                   dest_col=dest_col,
                   col_names=col_names,
                   mapping=mapping)


if __name__ == '__main__':

    np.random.seed(123)
    matrix = np.random.randn(4, 4, 2)

    # print(matrix)
    # skim = Skim(matrix, mapping=[2, 4, 6, 8], col_names=['time', 'dist'])
    #
    # # print(skim.to_numpy())
    # df = skim.to_dataframe()
    # print(df)

    sqlite_file = 'ambag_example/data/example.db'
    # sqlite_file = 'new_path_coef.db'
    # partial_mapping = [649, 652, 658, 660, 661, 663, 690, 691, 710, 714, 736, 741, 759, 1079, 1080, 1081, 1084, 1085, 1088, 1126]
    # skim = Skim.from_sqlite(sqlite_file, 'auto_skim', 'i', 'j', mapping=partial_mapping)

    skim = Skim.from_csv('ambag_example/data/nhbtrip.csv', 'ataz', 'ptaz')
    print(type(skim))
    df = skim.to_dataframe()
    print(df.head())
    print(df.shape)
    print(df.index)

    # new_skim = Skim(df, mapping=[3, 5, 6, 1])
    # print(new_skim.to_dataframe())
    # skim = Skim.from_sqlite()
    # print(skim.to_numpy())
