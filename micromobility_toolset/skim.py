import sqlite3

import numpy as np
import pandas as pd

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class Skim:
    def __init__(self, data, **kwargs):
        """
        data: numpy array or pandas DataFrame
        """

        if isinstance(data, np.ndarray):
            self.from_numpy(data, **kwargs)

        elif isinstance(data, pd.DataFrame):
            self.from_dataframe(data, **kwargs)

        else:
            raise TypeError("data must be a numpy array or pandas DataFrame")

    def from_numpy(
        self, data, mapping=None, orig_name=None, dest_name=None, core_names=None
    ):
        """
        data: 2- or 3- dimensional numpy array
        mapping: listlike of matrix index ids (int)
        orig_name: str, name of 1st dimension
        dest_name: str, name of 2nd dimension
        core_names: list of str, names for higher dimensions
        """

        self._set_matrix(data)
        self._set_mapping(mapping)
        self._set_num_cores()
        self._set_index(orig_name, dest_name)
        self._set_core_names(core_names)

    def from_dataframe(
        self, data, mapping=None, orig_name=None, dest_name=None, core_names=None
    ):

        if isinstance(data.index, pd.MultiIndex):
            # FIX: what if the index names already exist?
            data.index.names = [orig_name, dest_name]

        else:
            # FIX: avoid expensive reindexing?
            data.set_index([orig_name, dest_name], inplace=True)

        if mapping:

            # TODO: logger debug difference between dataframe and mapping
            # only retrieve rows from mapping
            data = data[
                data.index.isin(mapping, level=0) & data.index.isin(mapping, level=1)
            ]

        o_vals = data.index.get_level_values(0)
        d_vals = data.index.get_level_values(1)

        if not mapping:

            mapping = sorted(list(set(list(o_vals) + list(d_vals))))

        # if matrix_df.shape[0] == 0:
        #     return np.array([])

        matrix_length = len(mapping)

        if core_names:
            data = data[core_names]

        else:
            core_names = list(data.columns)

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

        if data.shape[1] > 1:
            np_matrix[o_mask, d_mask, :] = data.iloc[:, 0:].to_numpy()
        else:
            np_matrix[o_mask, d_mask] = data.iloc[:, 0].to_numpy()

        self._set_matrix(np_matrix)
        self._set_mapping(mapping)
        self._set_num_cores()
        self._set_index(orig_name, dest_name)
        self._set_core_names(core_names)

    def _set_matrix(self, data):
        if data.ndim not in [2, 3]:
            raise IndexError(f"input matrix must be 2 or 3 dimensions, not {data.ndim}")

        if not data.shape[0] == data.shape[1]:
            raise IndexError(f"matrix dimensions 1 and 2 do not match: {data.shape}")

        length = data.shape[0]
        if data.ndim == 2:
            data = data.reshape(length, length, 1)

        self._matrix = data
        self._length = length

    def _set_mapping(self, mapping):

        # TODO: make sure map is all ints
        if not mapping:
            self._mapping = np.arange(self._length)

        elif isinstance(mapping, list) or isinstance(mapping, np.ndarray):
            if not len(mapping) == self._length:
                raise IndexError(
                    f"mapping of {len(mapping)} items cannot be applied to matrix "
                    f"with shape {self._matrix.shape}"
                )

            self._mapping = np.array(mapping)

        else:
            raise TypeError("int or list for now")

    def _set_num_cores(self):

        if self._matrix.ndim == 2:
            self._num_cores = 1
        else:
            self._num_cores = self._matrix.shape[2]

    def _set_index(self, orig_name, dest_name):

        self._orig_name = orig_name
        self._dest_name = dest_name

    def _set_core_names(self, core_names):

        if not core_names:
            self._core_names = None

        elif isinstance(core_names, list):
            assert len(core_names) == self._num_cores
            self._core_names = core_names

        else:
            raise TypeError("None or list for now")

    @property
    def shape(self):

        return self._matrix.shape

    @property
    def length(self):

        return self._length

    @property
    def core_names(self):

        return self._core_names

    def get_core(self, name):

        if name not in self._core_names:
            raise KeyError(f"'{name} not found in skim")

        idx = self._core_names.index(name)

        return self._matrix[:, :, idx]

    def add_core(self, matrix, name):

        assert name not in self._core_names
        assert matrix.shape == (self._length, self._length) or matrix.shape == (
            self._length,
            self.length,
            1,
        )

        matrix = matrix.reshape((self._length, self._length, 1))
        new_matrix = np.concatenate((self._matrix, matrix), axis=2)

        self._set_matrix(new_matrix)
        self._set_num_cores()
        core_names = self._core_names + [name]
        self._set_core_names(core_names)

    def to_numpy(self):

        return self._matrix

    def to_dataframe(self):

        multi_index = pd.MultiIndex.from_arrays(
            [
                np.repeat(self._mapping, self._length),
                np.tile(self._mapping, self._length),
            ],
            names=[self._orig_name, self._dest_name],
        )

        # row-wise reshape, origins first, then destinations
        data = self._matrix.reshape(self._length ** 2, self._num_cores)

        df = pd.DataFrame(data, index=multi_index, columns=self._core_names)

        return df.loc[(df != 0).any(axis=1)]  # don't use all-zero rows

    def to_omx(self, filename):
        # need to check to make sure matrix entries are floats
        # also need to close file with a finally
        pass

    def to_sqlite(self, filename, table_name, **kwargs):

        db_connection = sqlite3.connect(filename)

        try:
            self.to_dataframe().to_sql(name=table_name, con=db_connection, **kwargs)

        finally:
            db_connection.close()

    def to_csv(self, filename, **kwargs):

        self.to_dataframe().to_csv(filename, **kwargs)

    @classmethod
    def from_omx(cls):
        pass

    @classmethod
    def from_sqlite(
        cls,
        sqlite_file,
        table_name,
        orig_name,
        dest_name,
        core_names=None,
        mapping=None,
    ):

        # open database cursor
        db_connection = sqlite3.connect(sqlite_file)

        try:
            matrix_df = pd.read_sql(
                f"select * from {table_name}",
                db_connection,
                index_col=[orig_name, dest_name],
                columns=core_names,
            )

            return cls(
                matrix_df,
                orig_name=orig_name,
                dest_name=dest_name,
                core_names=core_names,
                mapping=mapping,
            )

        finally:
            db_connection.close()

    @classmethod
    def from_csv(cls, csv_file, orig_name, dest_name, core_names=None, mapping=None):

        if core_names:
            # usecols doesn't include index_col values by default
            columns = list(core_names) + list([orig_name, dest_name])
        else:
            # None will include all columns
            columns = None

        matrix_df = pd.read_csv(
            csv_file, index_col=[orig_name, dest_name], usecols=None
        )

        return cls(
            matrix_df,
            orig_name=orig_name,
            dest_name=dest_name,
            core_names=core_names,
            mapping=mapping,
        )
