import random
from math import atan2, pi
import heapq

import sqlite3
import numpy as np
import pandas as pd


PREPROCESSORS = []

def preprocessor():
    
    def decorator(func):

        global PREPROCESSORS

        PREPROCESSORS.append(func)

    return decorator


class Network():

    def __init__(self, **kwargs):
        """initialize network data structure, void"""
        
        self.node_x_name = kwargs.get('node_x_name')
        self.node_y_name = kwargs.get('node_y_name')
        
        self.node_df = self.read_nodes(
            kwargs.get('node_file'),
            kwargs.get('node_table_name'),
            kwargs.get('node_name'),
            kwargs.get('node_attributes')
        )

        self.nodes = dict(zip(self.node_df.index, self.node_df.to_numpy().tolist()))
        self.node_names = list(self.node_df.columns)

        self.link_df = self.read_links(
            kwargs.get('link_file'),
            kwargs.get('link_table_name'),
            kwargs.get('from_name'),
            kwargs.get('to_name'),
            kwargs.get('link_attributes_by_direction')
        )

        self.create_adjacency_dict()
        self.centroid_connector_name = kwargs.get('centroid_connector_name')
        self.add_link_attribute(self.centroid_connector_name)
        self.check_network_completeness()
        self.create_dual()

        if PREPROCESSORS:
            for func in PREPROCESSORS:
                print(f'running {func.__name__}')
                func(self)
                print('done.')

    def read_nodes(self, file_path, table_name, node_name, attributes):
        """read links from sqlite database into network data structure, void

        file_path : name of link file
        node_name : column name of node id
        attributes : dictionary of { name in network data structure : name in database } """

        columns = list(attributes.values()) + [node_name]

        if file_path.endswith('.csv'):
            node_df = pd.read_csv(
                file_path,
                index_col=node_name,
                usecols=columns)

        elif file_path.endswith('.db'):
            db_connection = sqlite3.connect(file_path)

            node_df = pd.read_sql(
                f'select * from {table_name}',
                db_connection,
                index_col=node_name,
                columns=columns)

            db_connection.close()

        else:
            raise TypeError(f"cannot read nodes from filetype {file_path}")

        name_map = {v: k for k, v in attributes.items()}
        node_df.rename(name_map, inplace=True)

        return node_df

    def read_links(self, file_path,
                   table_name,
                   from_name,
                   to_name,
                   attributes_by_direction):

        """read links from sqlite database into network data structure, void

        file_path : path to csv
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

        columns = ab_columns + ba_columns + [from_name, to_name]

        if file_path.endswith('.csv'):
            link_df = pd.read_csv(file_path, usecols=columns)

        elif file_path.endswith('.db'):
            db_connection = sqlite3.connect(file_path)

            link_df = pd.read_sql(f'select * from {table_name}',
                                  db_connection,
                                  columns=columns)

            db_connection.close()

        else:
            raise TypeError(f'cannot read nodes from filetype {file_path}')

        ab_df = link_df[[from_name, to_name]].copy()
        ba_df = link_df[[from_name, to_name]].copy()

        # set directional column values
        for k, v in attributes_by_direction.items():
            ab_df[k] = link_df[v[0]]
            ba_df[k] = link_df[v[1]]

        # TODO: add a two_way network property
        ba_df.rename(columns={from_name: to_name, to_name: from_name}, inplace=True)

        return pd.concat([ab_df, ba_df], sort=True).set_index([from_name, to_name])

    def create_adjacency_dict(self):
        # nested dict of nodes, with first level being origin nodes, second destination
        # e.g. {0: {0: [34, 'TWO LANE']},
        #       2: {0: [23, 'FREEWAY']},
        #       3: {1: [45, 'RAMP'],
        #           2: [56, 'TWO LANE']}, ... }
        self.adjacency = {node: {} for node in self.link_df.index.get_level_values(0)}

        for nodes, vals in zip(self.link_df.index, self.link_df.values):
            self.adjacency[nodes[0]][nodes[1]] = list(vals)

        # put desired attribute names into network data structure
        self.adjacency_names = list(self.link_df.columns)

    def check_network_completeness(self):
        """check to see that all nodes have links and nodes for all links have defined attributes

        """

        node_nodes = set(self.node_df.index.values)
        link_nodes = set(
            list(self.link_df.index.get_level_values(0)) +
            list(self.link_df.index.get_level_values(1)))

        stray_nodes = node_nodes - link_nodes
        missing_nodes = link_nodes - node_nodes

        if stray_nodes:
            self.node_df = self.node_df[~self.node_df.index.isin(list(stray_nodes))]
            print(f'removed {len(stray_nodes)} stray nodes from network')

        if missing_nodes:
            raise Exception(f'missing {len(missing_nodes)} nodes from network: {missing_nodes}')

    def get_link_attribute_value(self,link,name):

        column_index = self.adjacency_names.index(name)
        return self.adjacency[link[0]][link[1]][column_index]

    def get_node_attribute_value(self,node,name):

        column_index = self.node_names.index(name)
        return self.nodes[node][column_index]

    def get_dual_attribute_value(self,link1,link2,name):

        column_index = self.dual_names.index(name)
        return self.dual[link1][link2][column_index]

    def set_link_attribute_value(self,link,name,value):

        column_index = self.adjacency_names.index(name)
        self.adjacency[link[0]][link[1]][column_index] = value

    def set_node_attribute_value(self,node,name,value):

        column_index = self.node_names.index(name)
        self.nodes[node][column_index] = value

    def set_dual_attribute_value(self,link1,link2,name,value):

        column_index = self.dual_names.index(name)
        self.dual[link1][link2][column_index] = value

    def add_link_attribute(self,name):

        assert name not in self.adjacency_names, f"{name} is already in adjacency list"

        self.adjacency_names.append(name)
        for a in self.adjacency:
            for b in self.adjacency[a]:
                self.adjacency[a][b].append(None)

    def add_node_attribute(self,name):

        assert name not in self.node_names, f"{name} is already in node list"

        self.node_names.append(name)
        for n in nodes:
            self.nodes[n].append(None)

    def add_dual_attribute(self,name):

        assert name not in self.dual_names, f"{name} is already in dual list"

        self.dual_names.append(name)
        for link1 in self.dual:
            for link2 in self.dual[link1]:
                self.dual[link1][link2].append(None)

    def set_centroid_connector_name(self,name):
        """set centroid connector field in name in node attributes, void"""

        self.centroid_connector_name = name

    def is_centroid_connector(self,link):
        """determine if an link is a centroid connector, boolean"""

        column_index = self.adjacency_names.index(self.centroid_connector_name)
        return self.adjacency[link[0]][link[1]][column_index]

    def link_angle(self,link1,link2):
        """return angular deviation traveling from link1 to link2, numeric"""

        # get index of x and y fields for node data
        x_column_index = self.node_names.index(self.node_x_name)
        y_column_index = self.node_names.index(self.node_y_name)

        # form xy vectors for links
        vector1 = ( (self.nodes[link1[0]][x_column_index],self.nodes[link1[0]][y_column_index]), (self.nodes[link1[1]][x_column_index],self.nodes[link1[1]][y_column_index]) )
        vector2 = ( (self.nodes[link2[0]][x_column_index],self.nodes[link2[0]][y_column_index]), (self.nodes[link2[1]][x_column_index],self.nodes[link2[1]][y_column_index]) )

        # calculate differences in x and y for vectors
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

    def traversal_type(self,link1,link2,consideration_field=None):
        """categorical indicator of maneuver required to traverse intersection over two links, numeric"""

        ## determination of left and right turns is made by min and max angle of deviation between links
        ## consideration_field is name of boolean link field which determines which are legs are considered in min and max calculation

        ## RETURN VALUES
        ## 0: from centroid connector to centroid connector
        ## 1: from centroid connector to street
        ## 2: from street to centroid connector
        ## 3: reversal
        ## 4: right onto unconsidered
        ## 5: left onto unconsidered
        ## 6: right at four-way or more
        ## 7: left at four-way or more
        ## 8: straight at four-way or more
        ## 9: right at T when straight not possible
        ## 10: left at T when straight not possible
        ## 11: right at T when left not possible
        ## 12: straight at T when left not possible
        ## 13: left at T when right not possible
        ## 14: straight at T when right not possible
        ## 15: continue with no other option


        # first identify if links connect
        if link1[1]!=link2[0]:
            raise Exception('cannot find traversal type if links do not connect')

        # then identify reversals and instances where one of the links is a centroid connector
        if self.is_centroid_connector(link1):
            if self.is_centroid_connector(link2):
                return 0
            else:
                return 1
        if self.is_centroid_connector(link2):
            return 2

        if link2[1]==link1[0]:
            return 3

        # get index for consideration field name
        if consideration_field is not None:
            consideration_column_index = self.adjacency_names.index(consideration_field)

        # get ready to find min and max angle
        min_angle = 999
        max_angle = -999
        count_legs = 0

        # for neighbors of the end of the first link
        for neighbor in self.adjacency[link1[1]]:

            # determine whether the neighbor should be considered
            consideration_flag = True
            if consideration_field is not None:
                consideration_flag = self.adjacency[link1[1]][neighbor][consideration_column_index]

            # if it should be considered and link is not a centroid connector or the same as the first link
            if consideration_flag and not ( self.is_centroid_connector((link1[1],neighbor)) or neighbor == link1[0] ):

                # calculate angle, update min and max and count of considered links
                current_angle = self.link_angle(link1,(link1[1],neighbor))
                min_angle= min(i for i in (min_angle,current_angle))
                max_angle = max(i for i in (max_angle,current_angle))
                count_legs = count_legs + 1

        # get angle of link we're determining the traversal type for
        this_angle = self.link_angle(link1,link2)

        if ( this_angle <  ( -  5.0 * pi / 6 )  ) or ( this_angle >  ( 5.0 * pi / 6 )  ):
            return 3
        if this_angle < min_angle:
            return 4
        if this_angle > max_angle:
            return 5

        # if it is a true intersection and the angle was the lowest, then right turn
        if count_legs >= 3:
            if this_angle <= min_angle:
                return 6
            if this_angle >= max_angle:
                return 7
            else:
                return 8
        if count_legs == 2:
            if min_angle < ( - pi / 4 ):
                if max_angle > ( pi / 4 ):
                    if this_angle <= min_angle:
                        return 9
                    else:
                        return 10
                else:
                    if this_angle <= min_angle:
                        return 11
                    else:
                        return 12
            else:
                if this_angle >= max_angle:
                    return 13
                else:
                    return 14
        else:
            return 15

    def node_degree_out(self,node,consideration_field=None):
        """number of links exiting intersection where consideration_field is True, numeric"""

        count_legs = 0

        for neighbor in self.adjacency[node]:

            consideration_flag = True
            if consideration_field is not None:
                consideration_flag = self.adjacency[node][neighbor][consideration_column_index]

            if consideration_flag and not self.is_centroid_connector((node,neighbor)):
                count_legs = count_legs + 1

        return count_legs

    def node_degree_in(self,node,consideration_field=None):
        """number of links entering intersection where consideration_field is True, numeric"""

        count_legs = 0

        for neighbor in self.adjacency[node]:

            consideration_flag = True
            if consideration_field is not None:
                consideration_flag = self.adjacency[neighbor][node][consideration_column_index]

            if consideration_flag and not self.is_centroid_connector((neighbor,node)):
                count_legs = count_legs + 1

        return count_legs


    def create_dual(self):
        """set up dual (link-to-link) adjacency for storing of turn and junction attributes, void"""

        self.dual = {}
        self.dual_names = []
        for node1 in self.adjacency:
            for node2 in self.adjacency[node1]:

                self.dual[(node1,node2)] = {}

                for node3 in self.adjacency[node2]:

                    link1 = (node1,node2)
                    link2 = (node2,node3)

                    self.dual[link1][link2] = []

    def single_source_dijkstra(self,source,variable_coefficients,target=None,randomization_scale=0,max_cost=None):
        """find lowest costs and paths between source and all nodes or target node with link cost cost_name and link-link cost dual_cost_name, tuple"""

        target_found = False
        if target is not None:
            if target not in self.adjacency:
                raise Exception('Target ' + str(target) + ' is not in network')

        if source == target:
            return ({source:0}, {source:[source]})

        dist = {} # dictionary of final costs
        paths = {} # dictionary of paths
        seen = {} #dictionary of visited nodes with temporary costs
        fringe = [] # priority queue heap of (cost, node) tuples

        # fill heap with initial node -> link traversals
        for neighbor in self.adjacency[source]:
            current_link = (source,neighbor)
            paths[current_link] = [(None,source),current_link]
            seen[current_link] = self.calculate_variable_cost(None,current_link,variable_coefficients,randomization_scale)
            heapq.heappush(fringe,(seen[current_link],current_link))

        # search remainder of graph with link -> link traversals
        while fringe:
            # pop link from queue, finalize distances, and check to see if target has been reached
            (from_dist,from_link) =  heapq.heappop(fringe)
            if from_link in dist:
                continue # already searched this node
            dist[from_link] = from_dist
            if from_link[1] == target:
                target_found = True
                break
            if max_cost is not None:
                if from_dist > max_cost:
                    break

            # iterate through neighbors
            for to_node in self.adjacency[from_link[1]]:
                to_link = (from_link[1],to_node)

                # calculate cost
                cur_dist = dist[from_link] +  self.calculate_variable_cost(from_link,to_link,variable_coefficients,randomization_scale)

                # add neighbor to heap and/or update temporary costs
                if to_link in dist:
                    if cur_dist < dist[to_link]:
                        raise ValueError('Contradictory paths found:','negative weights?')
                elif to_link not in seen or cur_dist < seen[to_link]:
                    seen[to_link] = cur_dist
                    heapq.heappush(fringe,(cur_dist,to_link))
                    paths[to_link] = paths[from_link] + [to_link]

        if target is not None and not target_found:
            raise Exception('Target ' + str(target) + ' was not found')

        # get node paths from list of links
        for p in paths:
            paths[p] = self.get_node_path_from_dual_path(paths[p])

        paths_nondual_keys = {}
        for p in paths:
            paths_nondual_keys[p[1]] = paths[p]
        paths = paths_nondual_keys

        dist_nondual_keys = {}
        for k in dist:
            dist_nondual_keys[k[1]] = dist[k]
        dist = dist_nondual_keys

        # return distances and paths
        return (dist,paths)

    def get_node_path_from_dual_path(self,p):
        for index in range(len(p)):
            p[index] = p[index][1]
        return p

    def calculate_variable_cost(self,from_link,to_link,variable_coefficients,randomization_scale):

        cost_value = 0

        for var in variable_coefficients:
            if var in self.adjacency_names:
                cost_value = cost_value + variable_coefficients[var] * self.get_link_attribute_value(to_link,var)
            elif var in self.dual_names:
                if from_link is None:
                    continue
                else:
                    cost_value = cost_value + variable_coefficients[var] * self.get_dual_attribute_value(from_link,to_link,var)
            else:
                raise Exception('variable name not found in network data')

        return (1.0 + random.random() * randomization_scale) * cost_value

    def get_nearby_pois(self, poi_ids, source_ids, varcoef, max_cost=None):
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

    def get_skim_matrix(self, node_ids, varcoef, max_cost=None):
        """skim network net starting from node_id to node_id, with variable coefficients varcoef
        until max_cost is reached, return matrix
        """

        num_nodes = len(node_ids)
        skim_matrix = np.zeros((num_nodes, num_nodes))

        for i, centroid in enumerate(node_ids):

            costs = self.single_source_dijkstra(centroid, varcoef, max_cost=max_cost)[0]

            for j, target in enumerate(node_ids):

                if target in costs:
                    skim_matrix[i, j] = costs[target]

        # Don't include intrazonal values
        return skim_matrix * (np.ones((num_nodes, num_nodes)) - np.diag(np.ones(num_nodes)))

    def get_attribute_matrix(self, attribute_name):
        """
        Gets matrix of attribute values for every node pair in the network
        """

        num_nodes = len(self.nodes)
        node_mapping = list(self.nodes.keys())
        matrix = np.zeros((num_nodes, num_nodes))

        for anode in self.adjacency:
            for bnode in self.adjacency[anode]:
                aidx = node_mapping.index(anode)
                bidx = node_mapping.index(bnode)
                matrix[aidx, bidx] = self.get_link_attribute_value((anode, bnode), attribute_name)

        return matrix

    def load_attribute_matrix(self, matrix, load_name, centroid_ids, varcoef, max_cost=None):
        """
        Add attribute values to a set of network links (links) given a list of node ids.

        Calculates every path between every node pair (until max cost) and adds attribute
        name/value to each intermediate link.
        """
        
        self.add_link_attribute(load_name)

        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.shape[0] == len(centroid_ids)

        for i, centroid in enumerate(centroid_ids):

            paths = self.single_source_dijkstra(centroid, varcoef, max_cost=max_cost)[1]

            for j, target in enumerate(centroid_ids):

                if target in paths and matrix[i,j] > 0:
                    for k in range(len(paths[target])-1):
                        link = (paths[target][k],paths[target][k+1])
                        prev = self.get_link_attribute_value(link,load_name) or 0
                        self.set_link_attribute_value(link,load_name,prev+matrix[i,j])
