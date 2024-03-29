import os
import sqlite3
import logging
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


def read_nodes(file_path, table_name, node_name, node_x_name, node_y_name):
    """read links from sqlite database into network data structure, void

    file_path : name of link file
    node_name : column name of node id
    node_x_name: column of node x coordinate
    node_y_name: column of node y coordinate
    """

    columns = [node_name, node_x_name, node_y_name]

    if file_path.endswith(".csv"):
        node_df = pd.read_csv(file_path, usecols=columns)

    elif file_path.endswith(".db"):
        db_connection = sqlite3.connect(file_path)

        node_df = pd.read_sql(
            f"select * from {table_name}", db_connection, columns=columns
        )

        db_connection.close()

    else:
        raise TypeError(f"cannot read nodes from filetype {file_path}")

    return node_df


def read_links(
    file_path, table_name, link_name, from_name, to_name, attributes_by_direction
):

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

    if file_path.endswith(".csv"):
        link_df = pd.read_csv(file_path, usecols=columns)

    elif file_path.endswith(".db"):
        db_connection = sqlite3.connect(file_path)

        link_df = pd.read_sql(
            f"select * from {table_name}", db_connection, columns=columns
        )

        db_connection.close()

    else:
        raise TypeError(f"cannot read links from filetype {file_path}")

    ab_df = link_df[[link_name, from_name, to_name]].copy()
    ba_df = link_df[[link_name, from_name, to_name]].copy()

    # set directional column values
    for k, v in attributes_by_direction.items():
        ab_df[k] = link_df[v[0]]
        ba_df[k] = link_df[v[1]]

    # TODO: add a two_way network property
    ba_df.rename(columns={from_name: to_name, to_name: from_name}, inplace=True)

    return pd.concat([ab_df, ba_df], sort=True)


def add_turn_edges(graph, node_x_name, node_y_name):
    """add helper edges to graph intersections with turn attributes

    edge attributes added:
    - turn, bool: whether or not an edge represents a turn
    - turn_type, str: either 'uturn', 'left', 'right', or 'straight'
    - parallel_aadt, float: AADT of outgoing edge
    - cross_aadt, float: max AADT at the intersection
    """

    graph.es["turn"] = False  # default value

    idxs = list(np.where(np.array(graph.outdegree()) > 2)[0])
    intersections = list(graph.vs[idxs])

    explode(graph, intersections)

    turns = graph.es.select(turn=True)

    # select source nodes for turns
    turn_node_idxs = list(set([edge.source for edge in turns]))
    turn_nodes = graph.vs[turn_node_idxs]

    # it's expensive to modify igraph object, so store all changes in lists and modify all at once
    # instantiate lists for each attribute that will be changed
    all_leg_idx = []
    all_turn_types = []
    all_aadt = []
    all_cross_aadt = []

    for node in turn_nodes:

        in_edges = node.in_edges()
        assert len(in_edges) == 1
        in_edge = in_edges[0]

        in_vector = xy_vector(in_edge.source_vertex, node, x=node_x_name, y=node_y_name)

        legs = node.out_edges()
        angles = []
        aadt = []

        for leg in legs:

            successors = leg.target_vertex.successors()
            assert len(successors) == 1

            out_vector = xy_vector(node, successors[0], x=node_x_name, y=node_y_name)
            angles.append(turn_angle(in_vector, out_vector))
            aadt.append(successors[0].out_edges()[0]["AADT"])  # TODO: parameterize

        min_angle = min(angles)
        max_angle = max(angles)

        turn_types = []

        for i, this_angle in enumerate(angles):

            if abs(this_angle) > (5.0 * pi / 6):
                turn_types.append("uturn")

            elif len(legs) >= 3:
                if this_angle == min_angle:
                    turn_types.append("right")

                elif this_angle == max_angle:
                    turn_types.append("left")

                else:
                    turn_types.append("straight")

            else:
                if min_angle < (-pi / 4):
                    if this_angle == min_angle:
                        turn_types.append("right")
                    else:
                        turn_types.append("left")
                else:
                    if this_angle == max_angle:
                        turn_types.append("left")
                    else:
                        turn_types.append("straight")
        
        all_leg_idx.append([leg.index for leg in legs])
        all_turn_types.append(turn_types)
        all_aadt.append(aadt)
        all_cross_aadt.append([max(aadt) * len(aadt)])

    # flatten lists
    all_leg_idx = [item for sublist in all_leg_idx for item in sublist]
    all_turn_types = [item for sublist in all_turn_types for item in sublist]
    all_aadt = [item for sublist in all_aadt for item in sublist]
    all_cross_aadt = [item for sublist in all_cross_aadt for item in sublist]

    # replace edge attributes in graph
    graph.es[all_leg_idx]["turn_type"] = all_turn_types
    graph.es[all_leg_idx]["parallel_aadt"] = all_aadt
    graph.es[all_leg_idx]["cross_aadt"] = all_cross_aadt


def explode(graph, nodes):
    """replace given nodes in graph with edges representing each
    incoming/outgoing pair
    """

    # note: this would need to be modified to accommodate an
    # undirected network graph

    edges_to_add = []
    edges_to_remove = []
    edges_to_replace = []
    replaced_edge_attrs = {attr: [] for attr in graph.edge_attributes()}

    for node in nodes:
        node_attrs = node.attributes()
        attrs = {k: node_attrs[k] for k in node_attrs if k not in ["name", "id"]}

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
    graph.add_edges(edges_to_add, attributes={"turn": True})


def xy_vector(node_1, node_2, x, y):

    return ((node_1[x], node_1[y]), (node_2[x], node_2[y]))


def turn_angle(vector1, vector2):

    xdiff1 = vector1[1][0] - vector1[0][0]
    xdiff2 = vector2[1][0] - vector2[0][0]
    ydiff1 = vector1[1][1] - vector1[0][1]
    ydiff2 = vector2[1][1] - vector2[0][1]

    angle = atan2(ydiff2, xdiff2) - atan2(ydiff1, xdiff1)

    if angle > pi:
        angle = angle - 2 * pi
    if angle < -pi:
        angle = 2 * pi + angle

    # return relative angle
    return angle


class Network:
    def __init__(self, **kwargs):
        """initialize network data structure, void"""

        self.name = kwargs.get("name", "Network")
        self.logger = logging.getLogger(self.name)

        self.node_name = kwargs["node_name"]
        self.node_x_name = kwargs["node_x_name"]
        self.node_y_name = kwargs["node_y_name"]
        self.link_name = kwargs["link_name"]
        self.link_from_node = kwargs["from_name"]
        self.link_to_node = kwargs["to_name"]

        self.node_df = read_nodes(
            kwargs["node_file"],
            kwargs.get("node_table_name"),
            kwargs["node_name"],
            self.node_x_name,
            self.node_y_name,
        )

        self.link_df = read_links(
            kwargs["link_file"],
            kwargs.get("link_table_name"),
            kwargs["link_name"],
            kwargs["from_name"],
            kwargs["to_name"],
            kwargs["link_attributes_by_direction"],
        )

        self.check_network_completeness()

        graph_file = kwargs.get("saved_graph", "")

        if os.path.exists(graph_file):

            self.logger.info(f"reading graph from {graph_file}")
            self._graph = ig.Graph.Read(graph_file)

        else:
            self._graph = self.create_igraph()

            if PREPROCESSORS:
                for func in PREPROCESSORS:
                    self.logger.info(f"running {func.__name__}")
                    func(self, kwargs)
                    self.logger.info("done.")

            if graph_file:

                self.logger.info(
                    f"saving graph to {graph_file}. "
                    f"move or delete this file to rebuild graph"
                )
                self._graph.write(graph_file)

    def check_network_completeness(self):
        """check to see that all nodes have links and nodes for all links have defined
        attributes
        """

        node_nodes = set(list(self.node_df[self.node_name]))
        link_nodes = set(
            list(self.link_df[self.link_from_node])
            + list(self.link_df[self.link_to_node])
        )

        stray_nodes = node_nodes - link_nodes
        missing_nodes = link_nodes - node_nodes

        if stray_nodes:
            # self.node_df = self.node_df[
            #     ~self.node_df[self.node_name].isin(list(stray_nodes))
            # ]
            # self.logger.info(f"removed {len(stray_nodes)} stray nodes from network")

            self.logger.info(f"network contains {len(stray_nodes)} stray nodes")

        if missing_nodes:
            raise Exception(
                f"missing {len(missing_nodes)} nodes from network: {missing_nodes}"
            )

    def create_igraph(self):
        """build graph representation of network"""

        # first two link columns need to be from/to nodes and
        # first node column must be node name.
        link_cols = list(self.link_df.columns)
        link_cols.insert(0, self.link_to_node)
        link_cols.insert(0, self.link_from_node)

        node_cols = list(self.node_df.columns)
        node_cols.remove(self.node_name)
        node_cols.insert(0, self.node_name)

        link_df = self.link_df[link_cols]
        node_df = self.node_df[node_cols]

        self.logger.info("building graph... ")
        graph = ig.Graph.DataFrame(edges=link_df, vertices=node_df, directed=True, use_vids=False)

        self.logger.info("done.")

        self.logger.info("adding turns... ")
        add_turn_edges(graph, self.node_x_name, self.node_y_name)

        self.logger.info("done.")

        return graph

    def get_edge_values(self, attr, dtype=None):

        if attr not in self._graph.edge_attributes():
            raise KeyError(f"{attr} is not an edge attribute")

        weights = np.array(self._graph.es[attr])

        nones = np.count_nonzero(weights == None)  # noqa

        if nones > 0:
            self.logger.debug(
                f"edge attribute {attr} contains {nones} missing values. "
                "replacing with NaNs."
            )

            weights[weights == None] = np.nan  # noqa

        return np.nan_to_num(weights.astype(dtype))

    def set_edge_values(self, attr, weights):

        if isinstance(weights, list) or isinstance(weights, np.ndarray):

            assert len(weights) == self._graph.ecount()

            self._graph.es[attr] = list(weights)
            return

        self._graph.es[attr] = weights

    def get_skim_matrix(self, node_ids, cost_attr, truncated=None, max_cost=None):
        """skim network net starting from node_id to node_id, using specified
        edge weights. Zero-out entries above max_cost, return matrix
        """

        weights = self.get_edge_values(cost_attr, dtype=np.float32)
        self.set_edge_values(cost_attr, np.nan_to_num(weights))

        # remove duplicate node_ids and save reconstruction indices
        node_ids = np.array(node_ids).astype(np.int64)
        nodes_uniq, node_map = np.unique(node_ids, return_inverse=True)

        vertex_names = np.array(self._graph.vs["name"])
        vertex_names = vertex_names[vertex_names != None].astype(np.int64)  # noqa

        assert np.isin(nodes_uniq, vertex_names).all(), "graph is missing some nodes"

        vertex_ids = np.searchsorted(vertex_names, nodes_uniq)

        costs = self._graph.shortest_paths(
            source=vertex_ids, target=vertex_ids, weights=cost_attr
        )

        # expand skim to match original node_ids
        skim_matrix = np.array(costs)[:, node_map][node_map, :]

        if max_cost:

            # when geenrating distance skims, create truncated matrix
            if cost_attr == "distance":
                truncated = skim_matrix > max_cost

            # otherwise, uss supplied matrix to truncate skims
            skim_matrix[truncated] = 0

        return skim_matrix.astype(np.float32), truncated # float 32 is precise enough

    def get_link_attributes(self, link_attrs):

        if not isinstance(link_attrs, list):

            link_attrs = [link_attrs]

        data = {
            self.link_from_node: self.get_edge_values(self.link_from_node),
            self.link_to_node: self.get_edge_values(self.link_to_node),
        }

        for attr in link_attrs:

            data[attr] = self.get_edge_values(attr)

        links_df = pd.DataFrame(data)
        links_df = links_df[(links_df != 0).any(axis=1)].dropna()
        links_df[[self.link_from_node, self.link_to_node]] = links_df[
            [self.link_from_node, self.link_to_node]
        ].astype(np.int64)

        return links_df.set_index([self.link_from_node, self.link_to_node])
