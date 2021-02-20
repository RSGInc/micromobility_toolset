"""micromobility_toolset.model

This module collects a list of model "steps" that each process some data from the given
configuration directories. A Model object manages the five directories that are used to run the
steps. See step().

A Model class instance is created at the start of a run and passed to each step. This way, steps
can share configuration settings and data with each other. Resources are stored as instance
attributes and methods, so it is easy to interact with the model when creating new steps.

Most of the model attributes (e.g. network_settings, base_network, zone_df) are only created
when called for the first time. This prevents the code from loading unecessary data into memory.
The model also tries to find existing versions of a resource instead of rebuilding it, whenever
possible. For example, the `base_bike_skim` and all other skim matrices are only pulled from the
network if a skim file is not found on disk.

The model also saves most commonly used resources (settings, networks, DataFrames) in a cache.
When a resource is requested, the cached version will be returned if it has been used before.

This means that when a resource is used in a step, the model will first look in the cache, then it
will look for it on disk, then finally will recalculate it as a last resort. If a series of steps
is run at once (via run([list, of, steps])), they will all share the same in-memory resources. If
they are run in separate python processes, they will need to reload/rebuild the data every time.

See ambag_bike_model.py for a usage example.

"""

import os
import time
import yaml
import logging
import warnings
import sqlite3

import numpy as np
import pandas as pd

from .skim import Skim
from .network import Network

STEPS = {}


def config_logger():

    # console handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # file handler
    logfile = f"micromobility_toolset_{time.strftime('%Y%b%d_%H_%M_%S_%p')}.log"
    fh = logging.FileHandler(logfile)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    logging.captureWarnings(True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[ch, fh],
        force=True)


def step():
    """
    Wrap each model step with the @step decorator.

    This will simply add the decorated function to this module's STEPS dictionary when the
    function is imported.
    """
    
    def decorator(func):
        name = func.__name__

        global STEPS
        assert name not in STEPS

        STEPS[name] = func

        return func

    return decorator


def list_steps():

    return list(STEPS.keys())


def run(steps, *scenarios):
    """
    Run a list of step names, in order. They will be run in the order they are given,
    so make sure the data dependencies make sense.
    """

    if not isinstance(steps, list):
        steps = [steps]

    if not all(step in STEPS for step in steps):
        raise KeyError(f'Invalid step list {steps}')

    config_logger()
    logger = logging.getLogger('Micromobility Toolset')

    for step in steps:
        start_time = time.perf_counter()
        logger.info(step.upper())
        step_func = STEPS.get(step)
        step_func(*scenarios)
        stop_time = time.perf_counter()
        logger.info(f'{step} completed in {stop_time - start_time:0.4f} seconds')


def _read_dataframe(file_path, index_col, table_name=None):

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, index_col=index_col)

    elif file_path.endswith('.db'):
        db_connection = sqlite3.connect(file_path)

        df = pd.read_sql(
            f"select * from {table_name}",
            db_connection,
            index_col=index_col)

        db_connection.close()

    else:
        raise TypeError(f'cannot read filetype {os.path.basename(file_path)}')

    return df


def filter_impact_area(base_scenario, build_scenario=None, zone_ids=None):

    logger = logging.getLogger('Impact Area')
    if not build_scenario:
        assert zone_ids is not None, \
            "need either a comparison scenario or a list of zone IDs"

        base_mask = base_scenario.zone_df.index.isin(zone_ids)

    else:
        zone_nodes = _get_zone_diff(base_scenario, build_scenario)

        base_node_col = base_scenario.zone_settings.get('zone_node_column')
        build_node_col = build_scenario.zone_settings.get('zone_node_column')
        base_mask = base_scenario.zone_df[base_node_col].isin(zone_nodes)
        build_mask = build_scenario.zone_df[build_node_col].isin(zone_nodes)

        # replace zone_dfs and related properties
        build_scenario.zone_df = build_scenario.zone_df[build_mask]

        del build_scenario.zone_nodes
        del build_scenario.zone_list

        logger.info(f'using {len(build_scenario.zone_df.index)} zones')

    base_scenario.zone_df = base_scenario.zone_df[base_mask]
    del base_scenario.zone_nodes
    del base_scenario.zone_list

    logger.warn(
        f'using {len(base_scenario.zone_df.index)} zones. '
        'make sure to delete output matrices if subsequent runs use more zones.')


def _get_zone_diff(base_scenario, build_scenario):
    base_zone_df = base_scenario.zone_df
    build_zone_df = build_scenario.zone_df

    for scenario, df in zip([base_scenario, build_scenario], [base_zone_df, build_zone_df]):
        scenario.logger.info(f'filtering {len(df.index)} zones...')

    changed_nodes = []

    base_links = base_scenario.network.link_df.round(4).reset_index()
    build_links = build_scenario.network.link_df.round(4).reset_index()
    link_diff = pd.concat([base_links, build_links]).drop_duplicates(keep=False)

    if link_diff.empty:
        base_scenario.logger.info(f'links match {build_scenario.name}')

    else:
        from_nodes = list(link_diff[base_scenario.network_settings.get('from_name')])
        to_nodes = list(link_diff[base_scenario.network_settings.get('to_name')])
        link_nodes = list(set(from_nodes + to_nodes))
        base_scenario.logger.info(f'links differ from {build_scenario.name} by {len(link_nodes)} nodes.')

        changed_nodes.extend(link_nodes)

    # 7 decimals of latlong precision is the practical limit of commercial surveying
    base_nodes = base_scenario.network.node_df.round(7).reset_index()
    build_nodes = build_scenario.network.node_df.round(7).reset_index()
    node_diff = pd.concat([base_nodes, build_nodes]).drop_duplicates(keep=False)

    if node_diff.empty:
        base_scenario.logger.info(f'nodes match {build_scenario.name}')

    else:
        nodes = list(set(node_diff[base_scenario.network_settings.get('node_name')]))
        base_scenario.logger.info(f'nodes differ from {build_scenario.name} by {len(nodes)} nodes.')

        changed_nodes.extend(nodes)

    node_col = base_scenario.zone_settings.get('zone_node_column')
    zone_diff = pd.concat([
        base_scenario.zone_df.round(4).reset_index(),
        build_scenario.zone_df.round(4).reset_index()]).drop_duplicates(keep=False)

    if zone_diff.empty:
        base_scenario.logger.info(f'zones match {build_scenario.name}')

    else:
        zone_nodes = list(set(zone_diff[node_col]))
        base_scenario.logger.info(f'zones differ from {build_scenario.name} by {len(zone_nodes)} zones.')

        changed_nodes.extend(zone_nodes)

    if not changed_nodes:
        return

    build_scenario.logger.info('getting nearby zones from the network...')
    nearby_zones = build_scenario.network.get_nearby_pois(
        build_zone_df[node_col],
        changed_nodes,
        build_scenario.network_settings.get('route_varcoef_bike'),
        max_cost=build_scenario.network_settings.get('max_cost_bike'))

    zone_nodes = []
    for zones in nearby_zones.values():
        zone_nodes.extend(zones)

    return list(set(zone_nodes))


class Scenario():

    def __init__(self, name, config, inputs, outputs):

        self._set_name(name)
        self._set_dirs(config, inputs, outputs)
        self._set_logger()

        # dictionary of property values
        self._cache = {}

        # dictionary of saved zone tables
        self._tables = {}

    def _set_name(self, name):

        self.name = name

    def _set_dirs(self, config, inputs, outputs):

        for path in [config, inputs]:
            assert os.path.isdir(path), f'Could not find directory {path}'

        if not os.path.isdir(outputs):
            os.mkdir(outputs)

        self._config_dir = config
        self._input_dir = inputs
        self._output_dir = outputs

    def _set_logger(self):

        self.logger = logging.getLogger(self.name)

    def config_file_path(self, filename):

        return os.path.join(self._config_dir, filename)

    def data_file_path(self, filename):
        """
        If user is providing a file from the input directory, use it.
        Otherwise write newly generated data to the output directory
        """

        data_path = os.path.join(self._input_dir, filename)

        if os.path.exists(data_path):
            return data_path

        else:
            return os.path.join(self._output_dir, filename)

    def cache(func):
        """
        Wrapper for the standard @property decorator with the additional
        perk of storing the return value in the instance's "_cache" dictionary.

        The function will be run normally the first time it is called. Subsequent
        calls will return the cached return value instead of calling the function
        again.

        Cached resources can be treated as normal instance attributes:

            `del resource` will remove it from the cache, causing its setter function to be
            called next time it is requested.

            resource = new_object will replace the cached attribute with new_object
        """
        name = func.__name__

        def get_prop(self):
            if name in self._cache:
                return self._cache[name]

            self.logger.debug(f"calling '{name}' for the first time")
            ret = func(self)
            self._cache[name] = ret

            return ret

        def set_prop(self, value):
            self.logger.debug(f"manually setting '{name}' in cache")
            self._cache[name] = value

        def del_prop(self):
            if name in self._cache:
                self.logger.debug(f"deleting '{name}' from cache")
                del self._cache[name]

            else:
                self.logger.debug(f"'del {name}' was called but '{name}' is not in cache") 

        return property(get_prop, set_prop, del_prop)

    def clear_cache(self):
        self.logger.debug('clearing cache')
        self._cache = {}

    def _read_settings_file(self, filename):

        file_path = self.config_file_path(filename)
        self.logger.debug(f'reading settings from {file_path}')

        with open(file_path) as f:
            settings = yaml.safe_load(f.read())

        return settings

    @cache
    def zone_settings(self):
        # TODO: dump into logger debug with some nice-ish
        # formatting along with network and trip settings
        return self._read_settings_file('zone.yaml')

    @cache
    def network_settings(self):
        return self._read_settings_file('network.yaml')

    @cache
    def trip_settings(self):
        return self._read_settings_file('trips.yaml')

    @cache
    def network(self):

        self.logger.info('creating network ...')
        net_settings = self.network_settings.copy()

        link_file = self.data_file_path(net_settings.get('link_file'))
        node_file = self.data_file_path(net_settings.get('node_file'))
        saved_graph = self.data_file_path(net_settings.get('saved_graph'))

        del net_settings['link_file']
        del net_settings['node_file']
        del net_settings['saved_graph']

        net = Network(
            link_file=link_file,
            node_file=node_file,
            saved_graph=saved_graph,
            **net_settings,
        )

        return net

    @cache
    def zone_df(self):
        file_name = self.zone_settings.get('zone_file_name')
        zone_col = self.zone_settings.get('zone_zone_column')
        node_col = self.zone_settings.get('zone_node_column')
        zone_table = self.zone_settings.get('zone_table_name')

        file_path = self.data_file_path(file_name)
        self.logger.debug(f'reading zones from {file_path}')

        zone_df = _read_dataframe(
            file_path, 
            zone_col,
            table_name=zone_table)

        self.logger.debug(f'zone dataframe columns: {zone_df.columns}')
        self.logger.debug(f'number of zones: {len(zone_df)}')

        return zone_df


    @cache
    def zone_nodes(self):
        nodes = self.zone_df[self.zone_settings.get('zone_node_column')]

        return list(nodes)

    @cache
    def zone_list(self):
        return list(self.zone_df.index)

    @cache
    def num_zones(self):

        self.logger.debug(f'num zones: {len(self.zone_list)}')
        return len(self.zone_list)

    @cache
    def reachable_zones(self):
        """tuple of (origin zone indices, destination zone indices) for non-intrazonal
        zone-to-zone pairs less than mode max dist
        """
        # TODO: parameterize mode

        return np.nonzero(self.bike_skim)

    @cache
    def zone_paths(self) :
        """nested dictionary of paths (lists of edge/link ids) between reachable zones.
        """
        # TODO: parameterize mode

        self.logger.info(f'calculating network paths for {len(self.reachable_zones[0])} zone pairs... ')

        zone_nodes = np.array(self.zone_nodes).astype(int)
        zone_array = np.array(self.zone_list).astype(int)

        paths = []
        for orig_idx in range(len(zone_nodes)):

            # skim indices of reachable destination zones
            dest_idxs = self.reachable_zones[1][np.where(self.reachable_zones[0]==orig_idx)[0]]
            orig_node = zone_nodes[orig_idx]  # note: zone_nodes have the same index as skim levels
            dest_nodes = zone_nodes[dest_idxs]

            # one-to-many shortest path search.
            # returns nested list of edge ids
            path_list = self.network.graph.get_shortest_paths(
                v=orig_node,
                to=dest_nodes,
                weights=self.network_settings.get('weights_bike'),  # see todo
                output='epath')

            paths.extend(path_list)

        self.logger.info('done.')
        return paths

    def _read_skim_file(self, file_path, table_name):

        ozone_col = self.network_settings.get('skim_pzone_col')
        dzone_col = self.network_settings.get('skim_azone_col')
        zones = self.zone_list

        file_name = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))

        # 3 dimensional matrix with time and distance
        self.logger.info(f'reading {file_name} ...')
        if file_path.endswith('.csv'):
            skim = Skim.from_csv(
                file_path,
                ozone_col, dzone_col,
                mapping=zones)

        elif file_path.endswith('.db'):
            skim = Skim.from_sqlite(
                file_path, table_name,
                ozone_col, dzone_col,
                mapping=zones)

        else:
            raise TypeError(f"cannot read skim from filetype {file_name}")

        return skim

    def _load_skim(self, mode):
        skim_file = self.network_settings.get(f'{mode}_skim_file')

        file_path = self.data_file_path(skim_file)

        skim_table = self.network_settings.get(f'{mode}_skim_table')
        ozone_col = self.network_settings.get('skim_pzone_col')
        dzone_col = self.network_settings.get('skim_azone_col')

        if os.path.exists(file_path):

            matrix = self._read_skim_file(file_path, skim_table).to_numpy()
            return matrix

        self.logger.info(f'skimming {mode} skim from network...')
        matrix = self.network.get_skim_matrix(
            self.zone_nodes,
            self.network_settings.get(f'weights_{mode}'),
            max_cost=self.network_settings.get(f'max_cost_{mode}'))

        skim = Skim(matrix,
                mapping=self.zone_list,
                orig_col=ozone_col,
                dest_col=dzone_col,
                col_names=[self.network_settings.get('skim_distance_col', 'distance')])

        if self.network_settings.get(f'save_{mode}_skim', False):
            self.logger.info(f'saving {mode} skim to {os.path.basename(os.path.dirname(file_path))}...')
            skim.to_csv(file_path)

        return skim.to_numpy()

    @cache
    def auto_skim(self):

        file_name = self.network_settings.get('auto_skim_file')
        table_name = self.network_settings.get('auto_skim_table')
        file_path = self.data_file_path(file_name)

        return self._read_skim_file(file_path, table_name).to_numpy()

    @cache
    def walk_skim(self):
        return self._load_skim('walk')

    @cache
    def bike_skim(self):
        return self._load_skim('bike')

    @cache
    def motorized_mode_indices(self):

        all_modes = self.trip_settings.get('modes')
        motorized_modes = self.trip_settings.get('motorized_modes')

        return [all_modes.index(mode) for mode in motorized_modes]

    @cache
    def bike_mode_indices(self):

        all_modes = self.trip_settings.get('modes')
        bike_modes = self.trip_settings.get('bike_modes')

        return [all_modes.index(mode) for mode in bike_modes]

    @cache
    def walk_mode_indices(self):

        all_modes = self.trip_settings.get('modes')
        walk_modes = self.trip_settings.get('walk_modes')

        return [all_modes.index(mode) for mode in walk_modes]

    def _read_zone_matrix(self, file_name, table_name=None):

        pzone_col = self.trip_settings.get('trip_pzone_col')
        azone_col = self.trip_settings.get('trip_azone_col')

        if file_name.endswith('.csv'):
            skim = Skim.from_csv(file_name, pzone_col, azone_col, mapping=self.zone_list)

        elif file_name.endswith('.db'):
            skim = Skim.from_sqlite(file_name, table_name,
                                pzone_col, azone_col,
                                mapping=self.zone_list)

        else:
            raise TypeError(f'cannot read matrix from filetype {os.path.basename(file_name)}')

        return skim.to_numpy()

    def load_util_matrix(self, segment):

        table_file = self.trip_settings.get('motorized_util_files').get(segment)

        return self._read_zone_matrix(self.data_file_path(table_file))

    def load_trip_matrix(self, segment):

        csv_file = self.trip_settings.get('trip_files').get(segment)

        # use CSVs for input if no sqlite db provided
        table_file = self.trip_settings.get('input_sqlite_db', csv_file)
        table_name = self.trip_settings.get('trip_tables', {}).get(segment)

        if segment in self._tables:
            self.logger.debug(f'returning cached {segment} trip matrix')
            return self._tables.get(segment)

        file_path = self.data_file_path(table_file)

        self.logger.debug(f'reading {segment} trip matrix from {file_path}')
        return self._read_zone_matrix(file_path, table_name=table_name)

    def save_trip_matrix(self, matrix, segment):

        table_file = self.trip_settings.get('trip_files').get(segment)

        self.logger.debug(f'caching {segment} trip matrix')
        self._tables[segment] = matrix

        col_names = self.trip_settings.get('modes')
        self.write_zone_matrix(matrix, table_file, col_names)

    def write_zone_matrix(self, matrix, filename, col_names):

        skim = Skim(matrix,
                mapping=self.zone_list,
                orig_col=self.trip_settings.get('trip_pzone_col'),
                dest_col=self.trip_settings.get('trip_azone_col'),
                col_names=col_names)

        filepath = self.data_file_path(filename)
        self.logger.debug(f'writing matrix with shape {matrix.shape} and headers {col_names} to {filename}')
        skim.to_csv(filepath)
