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

        # FIX: use data.ndim here
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

        if mapping:

            # only retrieve rows from mapping
            data = data[data.index.isin(mapping, level=0) & data.index.isin(mapping, level=1)]

        o_vals = data.index.get_level_values(0)
        d_vals = data.index.get_level_values(1)

        if not mapping:

            mapping = sorted(list(set(list(o_vals) + list(d_vals))))

        # if matrix_df.shape[0] == 0:
        #     return np.array([])

        matrix_length = len(mapping)

        if col_names:
            data = data[col_names]

        else:
            col_names = list(data.columns)

        # FIX: use data.ndim here
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

        o_mask = np.searchsorted(mapping, o_vals)
        d_mask = np.searchsorted(mapping, d_vals)

        # FIX: use data.ndim here
        if data.shape[1] > 1:
            np_matrix[o_mask, d_mask, :] = data.iloc[:, 0:].to_numpy()
        else:
            np_matrix[o_mask, d_mask] = data.iloc[:, 0].to_numpy()

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

        return df.loc[(df != 0).any(axis=1)]  # don't use all-zero rows

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
