import os
import sqlite3
import numpy as np
import pandas as pd
import igraph as ig


PREPROCESSORS = []

def preprocessor():

    def decorator(func):

        global PREPROCESSORS

        PREPROCESSORS.append(func)

    return decorator


def read_nodes(file_path, table_name, node_name, attributes):
    """read links from sqlite database into network data structure, void

    file_path : name of link file
    node_name : column name of node id
    attributes : dictionary of { name in network data structure : name in database } """

    columns = list(attributes.values()) + [node_name]

    if file_path.endswith('.csv'):
        node_df = pd.read_csv(
            file_path,
            usecols=columns)

    elif file_path.endswith('.db'):
        db_connection = sqlite3.connect(file_path)

        node_df = pd.read_sql(
            f'select * from {table_name}',
            db_connection,
            columns=columns)

        db_connection.close()

    else:
        raise TypeError(f"cannot read nodes from filetype {file_path}")

    name_map = {v: k for k, v in attributes.items()}
    node_df.rename(name_map, inplace=True)

    return node_df


def read_links(file_path,
               table_name,
               link_name,
               from_name,
               to_name,
               attributes_by_direction):

    """read links from sqlite database into network data structure, void

    file_path : path to csv
    link_name: link_id
    from_name : column name of from node
    to_name : column name of to node
    attributes_by_direction : dictionary of
        { name in network data structure : ( column name for ab direction,
                                             column name for ba direction) }

    """

    ab_columns = []
    ba_columns = []
    for ab, ba in attributes_by_direction.values():
        ab_columns.append(ab)
        ba_columns.append(ba)

    columns = ab_columns + ba_columns + [link_name, from_name, to_name]

    if file_path.endswith('.csv'):
        link_df = pd.read_csv(file_path, usecols=columns)

    elif file_path.endswith('.db'):
        db_connection = sqlite3.connect(file_path)

        link_df = pd.read_sql(f'select * from {table_name}',
                              db_connection,
                              columns=columns)

        db_connection.close()

    else:
        raise TypeError(f'cannot read links from filetype {file_path}')

    ab_df = link_df[[link_name, from_name, to_name]].copy()
    ba_df = link_df[[link_name, from_name, to_name]].copy()

    # set directional column values
    for k, v in attributes_by_direction.items():
        ab_df[k] = link_df[v[0]]
        ba_df[k] = link_df[v[1]]

    # TODO: add a two_way network property
    ba_df.rename(columns={from_name: to_name, to_name: from_name}, inplace=True)

    return pd.concat([ab_df, ba_df], sort=True)


class Network():

    def __init__(self, **kwargs):
        """initialize network data structure, void"""

        self.node_name = kwargs.get('node_name')
        self.link_name = kwargs.get('link_name')
        self.link_from_node = kwargs.get('from_name')
        self.link_to_node = kwargs.get('to_name')

        self.node_df = read_nodes(
            kwargs.get('node_file'),
            kwargs.get('node_table_name'),
            kwargs.get('node_name'),
            kwargs.get('node_attributes')
        )

        self.link_df = read_links(
            kwargs.get('link_file'),
            kwargs.get('link_table_name'),
            kwargs.get('link_name'),
            kwargs.get('from_name'),
            kwargs.get('to_name'),
            kwargs.get('link_attributes_by_direction')
        )

        self.check_network_completeness()

        self.graph = self.create_igraph(kwargs.get('saved_graph'))

        if PREPROCESSORS:
            for func in PREPROCESSORS:
                print(f'running {func.__name__}')
                func(self)
                print('done.')

    def check_network_completeness(self):
        """check to see that all nodes have links and nodes for all links have defined attributes

        """

        node_nodes = set(list(self.node_df[self.node_name]))
        link_nodes = set(
            list(self.link_df[self.link_from_node]) +
            list(self.link_df[self.link_to_node]))

        stray_nodes = node_nodes - link_nodes
        missing_nodes = link_nodes - node_nodes

        if stray_nodes:
            self.node_df = self.node_df[~self.node_df[self.node_name].isin(list(stray_nodes))]
            print(f'removed {len(stray_nodes)} stray nodes from network')

        if missing_nodes:
            raise Exception(f'missing {len(missing_nodes)} nodes from network: {missing_nodes}')

    def create_igraph(self, graph_file=None):
        """build graph representation of network
        """

        if os.path.exists(graph_file or ''):

            print(f'reading graph from {graph_file}')
            return ig.Graph.Read(graph_file)

        # first two link columns need to be from/to nodes and
        # first node column must be node name.
        # igraph expects vertex ids to be strings
        self.link_df[[self.link_from_node, self.link_to_node]] =\
            self.link_df[[self.link_from_node, self.link_to_node]].astype(str)

        self.node_df[self.node_name] = self.node_df[self.node_name].astype(str)

        link_df = self.link_df[[self.link_from_node, self.link_to_node, *self.link_df.columns]]
        node_df = self.node_df[[self.node_name, *self.node_df.columns]]

        graph = ig.Graph.DataFrame(
            edges=link_df,
            vertices=node_df,
            directed=True)

        if graph_file:

            print(f'saving graph to {graph_file}. move or delete this file to rebuild graph')
            graph.write(graph_file)

        return graph

    def get_skim_matrix(self, node_ids, weights, max_cost=None):
        """skim network net starting from node_id to node_id, using specified
        edge weights. Zero-out entries above max_cost, return matrix
        """

        # remove duplicate node_ids
        nodes_uniq = list(set(list(map(int, node_ids))))

        dists = self.graph.shortest_paths(
            source=nodes_uniq,
            target=nodes_uniq,
            weights=weights)

        # expand skim to match original node_ids
        node_map = [nodes_uniq.index(int(n)) for n in node_ids]
        skim_matrix = np.array(dists)[:, node_map][node_map, :]

        if max_cost:
            skim_matrix[skim_matrix > max_cost] = 0

        return skim_matrix

    # TODO: update this to igraph
    def get_nearby_pois(self, poi_ids, source_ids, weights, max_cost=None):
        """
        Gets list of nearby nodes for each source node.

        poi_ids: point-of-interest node ids to include in output values
        source_ids: output dictionary keys

        """

        nearby_pois = {}
        poi_set = set(poi_ids)

        for source in source_ids:
            paths = self.single_source_dijkstra(
                source,
                varcoef,
                max_cost=max_cost)[1]

            nearby_nodes = []
            for nodes in paths.values():
                nearby_nodes.extend(nodes)

            nearby_pois[source] = list(set(nearby_nodes) & poi_set)

        return nearby_pois

    def load_path_attributes(self, paths, attributes, load_name):
        """
        Sum attribute values over a list of paths (edge ids)

        Adds new edge attribute to all graph edges and sums up the cumulative
        attribute value for each intermediate link.
        """

        self.graph.es[load_name] = 0

        assert len(paths) == len(attributes)

        for i, attr in enumerate(attributes):

            path = paths[i]
            prev = np.array(self.graph.es[path][load_name])
            self.graph.es[path][load_name] = list(prev + attr)

    def get_link_attributes(self, link_attrs):

        if not isinstance(link_attrs, list):
            
            link_attrs = [link_attrs]
        # add new column to link_df

        data = {
            self.link_from_node: self.graph.es[self.link_from_node],
            self.link_to_node: self.graph.es[self.link_to_node]}

        for attr in link_attrs:
            
            data[attr] = self.graph.es[attr]

        return pd.DataFrame(data).set_index([self.link_from_node, self.link_to_node])
