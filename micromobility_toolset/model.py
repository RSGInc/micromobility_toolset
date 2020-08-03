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
import yaml
import sqlite3
import numpy as np
import pandas as pd

from .skim import Skim
from .network import Network

STEPS = {}


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

    for step in steps:
        print(f"\n{step.upper()}")
        step_func = STEPS.get(step)
        step_func(*scenarios)


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

    
    base_scenario.zone_df = base_scenario.zone_df[base_mask]
    del base_scenario.zone_nodes
    del base_scenario.zone_list

    print(f'using {len(base_scenario.zone_df.index)} {base_scenario.name} zones')


def _get_zone_diff(base_scenario, build_scenario):
    base_zone_df = base_scenario.zone_df
    build_zone_df = build_scenario.zone_df
    
    print(
        f'filtering {len(base_zone_df.index)} {base_scenario.name} zones',
        f'and {len(build_zone_df.index)} {build_scenario.name} zones...')
    
    changed_nodes = []

    base_links = base_scenario.network.link_df.round(4).reset_index()
    build_links = build_scenario.network.link_df.round(4).reset_index()
    link_diff = pd.concat([base_links, build_links]).drop_duplicates(keep=False)

    if link_diff.empty:
        print('base and build links are the same.')

    else:
        from_nodes = list(link_diff[base_scenario.network_settings.get('from_name')])
        to_nodes = list(link_diff[base_scenario.network_settings.get('to_name')])
        link_nodes = list(set(from_nodes + to_nodes))
        print(f'base and build links differ by {len(link_nodes)} nodes.')

        changed_nodes.extend(link_nodes)

    # 7 decimals of latlong precision is the practical limit of commercial surveying
    base_nodes = base_scenario.network.node_df.round(7).reset_index()
    build_nodes = build_scenario.network.node_df.round(7).reset_index()
    node_diff = pd.concat([base_nodes, build_nodes]).drop_duplicates(keep=False)

    if node_diff.empty:
        print('base and build nodes are the same.')

    else:
        nodes = list(set(node_diff[base_scenario.network_settings.get('node_name')]))
        print(f'base and build nodes differ by {len(nodes)} nodes.')

        changed_nodes.extend(nodes)

    node_col = base_scenario.zone_settings.get('zone_node_column')
    zone_diff = pd.concat([
        base_scenario.zone_df.round(4).reset_index(),
        build_scenario.zone_df.round(4).reset_index()]).drop_duplicates(keep=False)

    if zone_diff.empty:
        print('base and build zones are the same')

    else:
        zone_nodes = list(set(zone_diff[node_col]))
        print(f'base and build zones differ by {len(zone_nodes)} zones')

        changed_nodes.extend(zone_nodes)
    
    if not changed_nodes:
        return

    print('getting nearby zones from the build network...')
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

    def __init__(self, name, config, data):

        self._set_name(name)
        self._set_dirs(config, data)

        # dictionary of property values
        self._cache = {}

        # dictionary of saved zone tables
        self._tables = {}

    def _set_name(self, name):
        
        self.name = name

    def _set_dirs(self, config, data):

        for path in [config, data]:
            assert os.path.isdir(path), f'Could not find directory {path}'

        self._config_dir = config
        self._data_dir = data

    def config_file_path(self, filename):

        return os.path.join(self._config_dir, filename)

    def data_file_path(self, filename):

        return os.path.join(self._data_dir, filename)

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
                # returning cached value for 'name'
                return self._cache[name]

            # calling 'name' for the first time
            ret = func(self)
            self._cache[name] = ret

            return ret

        def set_prop(self, value):
            self._cache[name] = value

        def del_prop(self):
            if name in self._cache:
                del self._cache[name]

        return property(get_prop, set_prop, del_prop)

    def clear_cache(self):
        self._cache = {}

    def _read_settings_file(self, filename):

        with open(self.config_file_path(filename)) as f:
            settings = yaml.safe_load(f.read())

        return settings

    @cache
    def zone_settings(self):
        return self._read_settings_file('zone.yaml')

    @cache
    def network_settings(self):
        return self._read_settings_file('network.yaml')

    @cache
    def trip_settings(self):
        return self._read_settings_file('trips.yaml')

    @cache
    def network(self):

        print(f'creating {self.name} network ...')
        net_settings = self.network_settings.copy()

        link_file = self.data_file_path(net_settings.get('link_file'))
        node_file = self.data_file_path(net_settings.get('node_file'))

        del net_settings['link_file']
        del net_settings['node_file']

        net = Network(
            link_file=link_file,
            node_file=node_file,
            **net_settings,
        )

        return net
    
    @cache
    def zone_df(self):
        file_name = self.zone_settings.get('zone_file_name')
        zone_col = self.zone_settings.get('zone_zone_column')
        node_col = self.zone_settings.get('zone_node_column')
        zone_table = self.zone_settings.get('zone_table_name')

        zone_df = _read_dataframe(
            self.data_file_path(file_name),
            zone_col,
            table_name=zone_table)
        
        # print(f'loaded {zone_df.shape[0]} zones')
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
        return len(self.zone_list)

    def _read_skim_file(self, file_path, table_name):

        azone_col = self.network_settings.get('skim_azone_col')
        pzone_col = self.network_settings.get('skim_pzone_col')
        zones = self.zone_list

        file_name = os.path.basename(file_path)
        dir_name = os.path.basename(os.path.dirname(file_path))

        # 3 dimensional matrix with time and distance
        print(f'reading {file_name} from {self.name}...')
        if file_path.endswith('.csv'):
            skim = Skim.from_csv(
                file_path,
                azone_col, pzone_col,
                mapping=zones)

        elif file_path.endswith('.db'):
            skim = Skim.from_sqlite(
                file_path, table_name,
                azone_col, pzone_col,
                mapping=zones)

        else:
            raise TypeError(f"cannot read skim from filetype {file_name}")

        return skim

    def _load_skim(self, mode):
        skim_file = self.network_settings.get(f'{mode}_skim_file')

        file_path = self.data_file_path(skim_file)

        skim_table = self.network_settings.get(f'{mode}_skim_table')
        azone_col = self.network_settings.get('skim_azone_col')
        pzone_col = self.network_settings.get('skim_pzone_col')

        if os.path.exists(file_path):

            skim = self._read_skim_file(file_path, skim_table)

        else:
            print(f'skimming {mode} skim from {self.name} network...')
            matrix = self.network.get_skim_matrix(
                self.zone_nodes,
                self.network_settings.get(f'route_varcoef_{mode}'),
                max_cost=self.network_settings.get(f'max_cost_{mode}'))

            skim = Skim(matrix,
                    mapping=self.zone_list,
                    orig_col=azone_col,
                    dest_col=pzone_col,
                    col_names=[self.network_settings.get('skim_distance_col', 'distance')])

            if self.network_settings.get(f'save_{mode}_skim', False):
                print(f'saving {mode} skim to {os.path.basename(os.path.dirname(file_path))}...')
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

        azone_col = self.trip_settings.get('trip_azone_col')
        pzone_col = self.trip_settings.get('trip_pzone_col')

        if file_name.endswith('.csv'):
            skim = Skim.from_csv(file_name, azone_col, pzone_col, mapping=self.zone_list)

        elif file_name.endswith('.db'):
            skim = Skim.from_sqlite(file_name, table_name,
                                azone_col, pzone_col,
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
            return self._tables.get(segment)

        file_path = self.data_file_path(table_file)

        return self._read_zone_matrix(file_path, table_name=table_name)

    def save_trip_matrix(self, matrix, segment):

        table_file = self.trip_settings.get('trip_files').get(segment)

        self._tables[segment] = matrix

        col_names = self.trip_settings.get('modes')
        self.write_zone_matrix(matrix, table_file, col_names)

    def write_zone_matrix(self, matrix, filename, col_names):

        skim = Skim(matrix,
                mapping=self.zone_list,
                orig_col=self.trip_settings.get('trip_azone_col'),
                dest_col=self.trip_settings.get('trip_pzone_col'),
                col_names=col_names)

        filepath = self.data_file_path(filename)
        skim.to_csv(filepath)

