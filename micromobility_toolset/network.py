import os
import sqlite3
from math import atan2, pi
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

def explode(graph, nodes):
    # note: this would need to be modified to accommodate an
    # undirected network graph

    edges_to_add = []
    edges_to_remove = []
    edges_to_replace = []
    replaced_edge_attrs = {attr: [] for attr in graph.edge_attributes()}

    for node in nodes:
        node_attrs = node.attributes()
        attrs = {k:node_attrs[k] for k in node_attrs if k not in ['name', 'id']}

        in_nodes = []
        out_nodes = []

        for edge in node.in_edges():
            # create a new node to reattach the in-edge to
            new_node = graph.add_vertex(**attrs)
            in_nodes.append(new_node)

            # replace the edge's target vertex with the new one
            # (can't actually change the target, so just copy and delete)
            source_idx = edge.source
            edge_attrs = edge.attributes()
            edges_to_remove.append(edge)

            edges_to_replace.append((source_idx, new_node.index))
            for attr, val in edge_attrs.items():
                replaced_edge_attrs[attr].append(val)



        for edge in node.out_edges():
            # create a new node to reattach the out-edge to
            new_node = graph.add_vertex(**attrs)
            out_nodes.append(new_node)

            # replace the edge's source vertex with the new one
            # (can't actually change the source, so just copy and delete)
            target_idx = edge.target
            edge_attrs = edge.attributes()
            edges_to_remove.append(edge)

            edges_to_replace.append((new_node.index, target_idx))
            for attr, val in edge_attrs.items():
                replaced_edge_attrs[attr].append(val)

        # add the "turn" edges
        for i_node in in_nodes:
            for o_node in out_nodes:
                edges_to_add.append((i_node.index, o_node.index))

        # node.delete()

    graph.delete_edges(edges_to_remove)
    graph.add_edges(edges_to_replace, attributes=replaced_edge_attrs)
    graph.add_edges(edges_to_add, attributes={'turn': True})

def turn_angle(vector1, vector2):

    xdiff1 = vector1[1][0] - vector1[0][0]
    xdiff2 = vector2[1][0] - vector2[0][0]
    ydiff1 = vector1[1][1] - vector1[0][1]
    ydiff2 = vector2[1][1] - vector2[0][1]

    angle = atan2(ydiff2,xdiff2) - atan2(ydiff1,xdiff1)

    if angle > pi:
        angle = angle - 2 * pi
    if angle < -pi:
        angle = 2 * pi + angle

    # return relative angle
    return angle

def xy_vector(node_1, node_2):

    return ((node_1['xcoord'], node_1['ycoord']), (node_2['xcoord'], node_2['ycoord']))


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
        self.add_turn_edges()

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

        print('building graph... ', end='')
        graph = ig.Graph.DataFrame(
            edges=link_df,
            vertices=node_df,
            directed=True)

        print('done.')

        # print('adding turns... ', end='')
        # add_turn_edges(graph)

        # print('done.')

        if graph_file:

            print(f'saving graph to {graph_file}. move or delete this file to rebuild graph')
            graph.write(graph_file)

        return graph

    def add_turn_edges(self):
        """add helper edges to intersections with turn attributes

        edge attributes added:
        - turn, bool: whether or not an edge represents a turn
        - turn_type, str: either 'uturn', 'left', 'right', or 'straight'
        - parallel_aadt, float: AADT of outgoing edge
        - cross_aadt, float: max AADT at the intersection
        """

        # self.graph.es['turn'] = False  # default value

        idxs = list(np.where(np.array(self.graph.outdegree()) > 2)[0])
        intersections = list(self.graph.vs[idxs])

        explode(self.graph, intersections)

        turns = self.graph.es.select(turn=True)

        # select source nodes for turns
        turn_node_idxs = list(set([edge.source for edge in turns]))
        turn_nodes = self.graph.vs[turn_node_idxs]

        for node in turn_nodes:

            in_edges = node.in_edges()
            assert len(in_edges) == 1
            in_edge = in_edges[0]

            in_vector = xy_vector(in_edge.source_vertex, node)

            legs = node.out_edges()
            angles = []
            aadt = []

            for leg in legs:

                successors = leg.target_vertex.successors()
                assert len(successors) == 1

                out_vector = xy_vector(node, successors[0])
                angles.append(turn_angle(in_vector, out_vector))
                aadt.append(successors[0].out_edges()[0]['AADT'])  # TODO: parameterize

            min_angle = min(angles)
            max_angle = max(angles)

            turn_types = []

            for i, this_angle in enumerate(angles):

                if ( abs(this_angle) > ( 5.0 * pi / 6 )  ):
                    turn_types.append('uturn')

                elif len(legs) >= 3:
                    if this_angle == min_angle:
                        turn_types.append('right')

                    elif this_angle == max_angle:
                        turn_types.append('left')

                    else:
                        turn_types.append('straight')

                else:
                    if min_angle < ( - pi / 4 ):
                        if this_angle == min_angle:
                            turn_types.append('right')
                        else:
                            turn_types.append('left')
                    else:
                        if this_angle == max_angle:
                            turn_types.append('left')
                        else:
                            turn_types.append('straight')

            leg_idxs = [leg.index for leg in legs]
            self.graph.es[leg_idxs]['turn_type'] = turn_types
            self.graph.es[leg_idxs]['parallel_aadt'] = aadt
            self.graph.es[leg_idxs]['cross_aadt'] = max(aadt)

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

        return pd.DataFrame(data).set_index([self.link_from_node, self.link_to_node]).dropna()
