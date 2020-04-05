import random
from math import atan2, pi
import heapq

import numpy as np
import pandas as pd

from activitysim.core.config import data_file_path


class Network():

    def __init__(self, network_settings):
        """initialize network data structure, void"""

        self.adjacency = {}
        self.nodes = {}
        self.dual = {}

        self.adjacency_names = []
        self.node_names = []
        self.dual_names = []

        self.node_x_name = None
        self.node_y_name = None

        self.centroid_connector_name = None

        self.read_links(
            network_settings.get('link_file'),
            network_settings.get('from_name'),
            network_settings.get('to_name'),
            network_settings.get('link_attributes_by_direction')
        )

        self.read_nodes(
            network_settings.get('node_file'),
            network_settings.get('node_name'),
            network_settings.get('node_attributes')
        )

        self.check_network_completeness()

        self.set_node_x_name(network_settings.get('node_x_name'))
        self.set_node_y_name(network_settings.get('node_y_name'))

        self.add_edge_attribute(network_settings.get('centroid_connector_name'))
        self.centroid_connector_name = network_settings.get('centroid_connector_name')
        centroid_connector_test = network_settings.get('centroid_connector_test')
        for a in self.adjacency:
            for b in self.adjacency[a]:
                if self.get_edge_attribute_value((a, b), centroid_connector_test[0]) == centroid_connector_test:
                    self.set_edge_attribute_value((a, b), self.centroid_connector_name, True)
                else:
                    self.set_edge_attribute_value((a, b), self.centroid_connector_name, False)

        self.create_dual()

    def read_links(self, file_name,
                   from_name,
                   to_name,
                   attributes_by_direction):

        """read links from sqlite database into network data structure, void

        file_name : path to csv
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

        file_name = data_file_path(file_name)
        link_df = pd.read_csv(file_name, usecols=ab_columns + ba_columns + [from_name, to_name])

        ab_df = link_df[[from_name, to_name]].copy()
        ba_df = link_df[[from_name, to_name]].copy()

        # set directional column values
        for k, v in attributes_by_direction.items():
            ab_df[k] = link_df[v[0]]
            ba_df[k] = link_df[v[1]]

        # TODO: add a two_way network property
        ba_df.rename(columns={from_name: to_name, to_name: from_name}, inplace=True)

        link_df = pd.concat([ab_df, ba_df]).set_index([from_name, to_name])

        # nested dict of nodes, with first level being origin nodes, second destination
        # e.g. {0: {0: [34, 'TWO LANE']},
        #       2: {0: [23, 'FREEWAY']},
        #       3: {1: [45, 'RAMP'],
        #           2: [56, 'TWO LANE']}, ... }
        self.adjacency = {node: {} for node in link_df.index.get_level_values(0)}

        for nodes, vals in zip(link_df.index, link_df.values):
            self.adjacency[nodes[0]][nodes[1]] = list(vals)

        # put desired attribute names into network data structure
        self.adjacency_names = list(link_df.columns)


    def read_nodes(self, file_name, node_name, attributes):
        """read links from sqlite database into network data structure, void

        file_name : name of link file
        node_name : column name of node id
        attributes : dictionary of { name in network data structure : name in database }
        """

        # put desired attribute names into network data structure
        self.node_names = list(attributes.keys())

        columns = list(attributes.values()) + [node_name]

        file_name = data_file_path(file_name)
        node_df = pd.read_csv(file_name,
                              index_col=node_name,
                              usecols=columns)

        self.nodes = dict(zip(node_df.index, node_df.to_numpy().tolist()))

    def check_network_completeness(self):
        """check to see that all nodes have edges and nodes for all edges have defined attributes

        """

        stray_nodes = []
        missing_nodes = []

        for node, vals in dict(self.nodes).items():
            if len(vals) == 0:
                # empty node was added by link reader
                missing_nodes.append(node)

            if node not in self.adjacency:
                # node does not have edge
                stray_nodes.append(node)
                del self.nodes[node]

        if stray_nodes:
            print('removed %s stray nodes from network' % len(stray_nodes))

        if missing_nodes:
            raise Exception('missing %s nodes from network: %s' % (len(missing_nodes), missing_nodes))

    def get_edge_attribute_value(self,edge,name):

        column_index = self.adjacency_names.index(name)
        return self.adjacency[edge[0]][edge[1]][column_index]

    def get_node_attribute_value(self,node,name):

        column_index = self.node_names.index(name)
        return self.nodes[node][column_index]

    def get_dual_attribute_value(self,edge1,edge2,name):

        column_index = self.dual_names.index(name)
        return self.dual[edge1][edge2][column_index]

    def set_edge_attribute_value(self,edge,name,value):

        column_index = self.adjacency_names.index(name)
        self.adjacency[edge[0]][edge[1]][column_index] = value

    def set_node_attribute_value(self,node,name,value):

        column_index = self.node_names.index(name)
        self.nodes[node][column_index] = value

    def set_dual_attribute_value(self,edge1,edge2,name,value):

        column_index = self.dual_names.index(name)
        self.dual[edge1][edge2][column_index] = value

    def add_edge_attribute(self,name):

        self.adjacency_names.append(name)
        for a in self.adjacency:
            for b in self.adjacency[a]:
                self.adjacency[a][b].append(None)

    def add_node_attribute(self,name):

        self.node_names.append(name)
        for n in nodes:
            self.nodes[n].append(None)

    def add_dual_attribute(self,name):

        self.dual_names.append(name)
        for edge1 in self.dual:
            for edge2 in self.dual[edge1]:
                self.dual[edge1][edge2].append(None)

    def set_node_x_name(self,name):
        """set node x name in node attributes, void"""

        self.node_x_name = name

    def set_node_y_name(self,name):
        """set node y name in node attributes, void"""

        self.node_y_name = name

    def set_centroid_connector_name(self,name):
        """set centroid connector field in name in node attributes, void"""

        self.centroid_connector_name = name

    def is_centroid_connector(self,edge):
        """determine if an edge is a centroid connector, boolean"""

        column_index = self.adjacency_names.index(self.centroid_connector_name)
        return self.adjacency[edge[0]][edge[1]][column_index]

    def edge_angle(self,edge1,edge2):
        """return angular deviation traveling from edge1 to edge2, numeric"""

        # get index of x and y fields for node data
        x_column_index = self.node_names.index(self.node_x_name)
        y_column_index = self.node_names.index(self.node_y_name)

        # form xy vectors for edges
        vector1 = ( (self.nodes[edge1[0]][x_column_index],self.nodes[edge1[0]][y_column_index]), (self.nodes[edge1[1]][x_column_index],self.nodes[edge1[1]][y_column_index]) )
        vector2 = ( (self.nodes[edge2[0]][x_column_index],self.nodes[edge2[0]][y_column_index]), (self.nodes[edge2[1]][x_column_index],self.nodes[edge2[1]][y_column_index]) )

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

    def traversal_type(self,edge1,edge2,consideration_field=None):
        """categorical indicator of maneuver required to traverse intersection over two edges, numeric"""

        ## determination of left and right turns is made by min and max angle of deviation between edges
        ## consideration_field is name of boolean edge field which determines which are legs are considered in min and max calculation

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


        # first identify if edges connect
        if edge1[1]!=edge2[0]:
            raise Exception('cannot find traversal type if edges do not connect')

        # then identify reversals and instances where one of the edges is a centroid connector
        if self.is_centroid_connector(edge1):
            if self.is_centroid_connector(edge2):
                return 0
            else:
                return 1
        if self.is_centroid_connector(edge2):
            return 2

        if edge2[1]==edge1[0]:
            return 3

        # get index for consideration field name
        if consideration_field is not None:
            consideration_column_index = self.adjacency_names.index(consideration_field)

        # get ready to find min and max angle
        min_angle = 999
        max_angle = -999
        count_legs = 0

        # for neighbors of the end of the first edge
        for neighbor in self.adjacency[edge1[1]]:

            # determine whether the neighbor should be considered
            consideration_flag = True
            if consideration_field is not None:
                consideration_flag = self.adjacency[edge1[1]][neighbor][consideration_column_index]

            # if it should be considered and edge is not a centroid connector or the same as the first edge
            if consideration_flag and not ( self.is_centroid_connector((edge1[1],neighbor)) or neighbor == edge1[0] ):

                # calculate angle, update min and max and count of considered edges
                current_angle = self.edge_angle(edge1,(edge1[1],neighbor))
                min_angle= min(i for i in (min_angle,current_angle))
                max_angle = max(i for i in (max_angle,current_angle))
                count_legs = count_legs + 1

        # get angle of edge we're determining the traversal type for
        this_angle = self.edge_angle(edge1,edge2)

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
        """number of edges exiting intersection where consideration_field is True, numeric"""

        count_legs = 0

        for neighbor in self.adjacency[node]:

            consideration_flag = True
            if consideration_field is not None:
                consideration_flag = self.adjacency[node][neighbor][consideration_column_index]

            if consideration_flag and not self.is_centroid_connector((node,neighbor)):
                count_legs = count_legs + 1

        return count_legs

    def node_degree_in(self,node,consideration_field=None):
        """number of edges entering intersection where consideration_field is True, numeric"""

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

        for node1 in self.adjacency:
            for node2 in self.adjacency[node1]:

                self.dual[(node1,node2)] = {}

                for node3 in self.adjacency[node2]:

                    edge1 = (node1,node2)
                    edge2 = (node2,node3)

                    self.dual[edge1][edge2] = []

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

        # fill heap with initial node -> edge traversals
        for neighbor in self.adjacency[source]:
            current_edge = (source,neighbor)
            paths[current_edge] = [(None,source),current_edge]
            seen[current_edge] = self.calculate_variable_cost(None,current_edge,variable_coefficients,randomization_scale)
            heapq.heappush(fringe,(seen[current_edge],current_edge))

        # search remainder of graph with edge -> edge traversals
        while fringe:
            # pop edge from queue, finalize distances, and check to see if target has been reached
            (from_dist,from_edge) =  heapq.heappop(fringe)
            if from_edge in dist:
                continue # already searched this node
            dist[from_edge] = from_dist
            if from_edge[1] == target:
                target_found = True
                break
            if max_cost is not None:
                if from_dist > max_cost:
                    break

            # iterate through neighbors
            for to_node in self.adjacency[from_edge[1]]:
                to_edge = (from_edge[1],to_node)

                # calculate cost
                cur_dist = dist[from_edge] +  self.calculate_variable_cost(from_edge,to_edge,variable_coefficients,randomization_scale)

                # add neighbor to heap and/or update temporary costs
                if to_edge in dist:
                    if cur_dist < dist[to_edge]:
                        raise ValueError('Contradictory paths found:','negative weights?')
                elif to_edge not in seen or cur_dist < seen[to_edge]:
                    seen[to_edge] = cur_dist
                    heapq.heappush(fringe,(cur_dist,to_edge))
                    paths[to_edge] = paths[from_edge] + [to_edge]

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

    def calculate_variable_cost(self,from_edge,to_edge,variable_coefficients,randomization_scale):

        cost_value = 0

        for var in variable_coefficients:
            if var in self.adjacency_names:
                cost_value = cost_value + variable_coefficients[var] * self.get_edge_attribute_value(to_edge,var)
            elif var in self.dual_names:
                if from_edge is None:
                    continue
                else:
                    cost_value = cost_value + variable_coefficients[var] * self.get_dual_attribute_value(from_edge,to_edge,var)
            else:
                raise Exception('variable name not found in network data')

        return (1.0 + random.random() * randomization_scale) * cost_value

    def path_trace(self,path,varname,tracefun='sum',finalfun=None,wgtvar=None):
        """tracefun is one of 'sum', 'min', 'max', 'avg'
        finalfun is one of None, 'inv' for inverse, 'neg' for negative
        wgtvar is the name of the variable for weighting sums, avgs, if desired"""

        L=len(path)-1
        sum_val=0
        min_val=None
        max_val=None
        wgt_sum=0

        if varname in self.dual_names:
            vardual = True
            var_index = self.dual_names.index(varname)
        else:
            vardual = False
            var_index = self.adjacency_names.index(varname)

        if wgtvar != None:

            if wgtvar in self.dual_names:
                wgtdual = True
                wgt_index = self.dual_names.index(wgtvar)
            else:
                wgtdual = False
                wgt_index = self.adjacency_names.index(wgtvar)

            if wgtdual != vardual:
                raise Exception('one of varname and wgtvar is a edge variable and the other is an edge pair variable')

        for i in range(L):
            wgt_val=1
            trace_val=0

            if vardual:
                if i < (L-1):
                    trace_val = self.dual[ (path[i],path[i+1]) ][ (path[i+1],path[i+2]) ][var_index]
                    if wgtvar is not None:
                        wgt_val = self.dual[ (path[i],path[i+1]) ][ (path[i+1],path[i+2]) ][wgt_index]
            else:
                trace_val = self.adjacency[ path[i] ][ path[i+1] ][var_index]
                if wgtvar is not None:
                    wgt_val = self.adjacency[ path[i] ][ path[i+1] ][wgt_index]

            trace_val=trace_val*wgt_val
            wgt_sum=wgt_sum+wgt_val
            sum_val=sum_val+trace_val
            min_val=min(min_val,trace_val)
            max_val=max(max_val,trace_val)

        if wgtvar is None:
            wgt_sum=L
        if tracefun=='sum':
            return_val=sum_val
        if tracefun=='min':
            return_val=min_val
        if tracefun=='max':
            return_val=max_val
        if tracefun=='avg':
            if wgt_sum==0 and sum_val==0:
                return_val=0
            else:
                return_val=sum_val/wgt_sum
        if finalfun=='inv':
            return 1/return_val
        if finalfun=='neg':
            return -return_val
        return return_val

    def get_path_sizes(self,path_list,overlap_var):

        temp=[]
        hashes=[]
        lengths=[]
        for cur_path in path_list:
            cur_hash={}
            cur_length=0
            for i in range(len(cur_path)-1):
                cur_edge = (cur_path[i],cur_path[i+1])
                cur_hash[cur_edge] = self.get_edge_attribute_value(cur_edge,overlap_var)
                cur_length = cur_length + cur_hash[cur_edge]
            hashes.append(cur_hash)
            lengths.append(cur_length)
        min_length = min(lengths)

        for i in range(len(path_list)):
            PS=0
            if lengths[i] > 0:
                for a in hashes[i]:
                    delta_sum=0
                    for j in range(len(path_list)):
                        if a in hashes[j]:
                            if min_length == 0:
                                delta_sum=delta_sum+1
                            else:
                                delta_sum=delta_sum+min_length/lengths[j]
                    PS=PS+hashes[i][a]/lengths[i]/delta_sum
            temp.append(PS)

        return temp

    def get_skim_matrix(self, taz_nodes, varcoef, max_cost=None):
        """skim network net starting from taz nodes in taz_nodes, with variable coefficients varcoef
        until max_cost is reached, return matrix
        """

        nzones = len(taz_nodes)
        skim_matrix = np.zeros((nzones, nzones))

        for i, ataz in enumerate(taz_nodes.keys()):

            centroid = taz_nodes[ataz]  # node id
            costs = self.single_source_dijkstra(centroid, varcoef, max_cost=max_cost)[0]

            for j, btaz in enumerate(taz_nodes.keys()):

                if taz_nodes[btaz] in costs:
                    skim_matrix[i, j] = costs[taz_nodes[btaz]]

        # Don't include intrazonal values
        return skim_matrix * (np.ones((nzones, nzones)) - np.diag(np.ones(nzones)))

    def get_attribute_matrix(self, attribute_name):

        num_nodes = len(self.nodes)
        matrix = np.zeros((num_nodes, num_nodes))

        for i, anode in enumerate(self.adjacency):
            for j, bnode in enumerate(self.adjacency[anode]):
                matrix[i, j] = self.get_edge_attribute_value((anode, bnode), attribute_name)

        return matrix

    def load_attribute_matrix(self, trips, load_name, taz_nodes, varcoef, max_cost=None):

        for i, ataz in enumerate(taz_nodes.keys()):

            centroid = taz_nodes[ataz]
            paths = self.single_source_dijkstra(centroid, varcoef, max_cost=max_cost)[1]

            for j, btaz in enumerate(taz_nodes.keys()):

                target = taz_nodes[btaz]

                if target in paths and trips[i,j] > 0:
                    for k in range(len(paths[target])-1):
                        edge = (paths[target][k],paths[target][k+1])
                        self.set_edge_attribute_value(edge,load_name,trips[i,j]+self.get_edge_attribute_value(edge,load_name))

    def add_derived_network_attributes(self, coef_walk=None, coef_bike=None):
        """add selfwork attributes that are combinations of existing attributes"""

        # add new link attribute columns
        self.add_edge_attribute('d0') # distance on ordinary streets, miles
        self.add_edge_attribute('d1') # distance on bike paths
        self.add_edge_attribute('d2') # distance on bike lanes
        self.add_edge_attribute('d3') # distance on bike routes
        self.add_edge_attribute('dne1') # distance not on bike paths
        self.add_edge_attribute('dne2') # distance not on bike lanes
        self.add_edge_attribute('dne3') # distance not on bike routes
        self.add_edge_attribute('dw') # distance wrong way
        #gain now comes directly from sqlite
        #self.add_edge_attribute('riseft')
        self.add_edge_attribute('auto_permit') # autos permitted
        self.add_edge_attribute('bike_exclude') # bikes excluded
        self.add_edge_attribute('dloc') # distance on local streets
        self.add_edge_attribute('dcol') # distance on collectors
        self.add_edge_attribute('dart') # distance on arterials
        self.add_edge_attribute('dne3loc') # distance on locals with no bike route
        self.add_edge_attribute('dne2art') # distance on arterials with no bike lane
        self.add_edge_attribute('bike_vol')

        # loop over edges and calculate derived values
        for a in self.adjacency:
            for b in self.adjacency[a]:
                distance = self.get_edge_attribute_value((a,b),'distance')
                bike_class = self.get_edge_attribute_value((a,b),'bike_class')
                lanes = self.get_edge_attribute_value((a,b),'lanes')
                #gain now comes directly from sqlite
                #from_elev = self.get_edge_attribute_value((a,b),'from_elev')
                #to_elev = self.get_edge_attribute_value((a,b),'to_elev')
                link_type = self.get_edge_attribute_value((a,b),'link_type')
                fhwa_fc = self.get_edge_attribute_value((a,b),'fhwa_fc')
                self.set_edge_attribute_value( (a,b), 'd0', distance * ( bike_class == 0 and lanes > 0 ) )
                self.set_edge_attribute_value( (a,b), 'd1', distance * ( bike_class == 1 ) )
                self.set_edge_attribute_value( (a,b), 'd2', distance * ( bike_class == 2 ) )
                self.set_edge_attribute_value( (a,b), 'd3', distance * ( bike_class == 3 ) )
                self.set_edge_attribute_value( (a,b), 'dne1', distance * ( bike_class != 1 ) )
                self.set_edge_attribute_value( (a,b), 'dne2', distance * ( bike_class != 2 ) )
                self.set_edge_attribute_value( (a,b), 'dne3', distance * ( bike_class != 3 ) )
                self.set_edge_attribute_value( (a,b), 'dw', distance * ( bike_class == 0 and lanes == 0 ) )
                #gain now comes directly from sqlite
                #self.set_edge_attribute_value( (a,b), 'riseft',  max(to_elev - from_elev,0) )
                self.set_edge_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
                self.set_edge_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
                self.set_edge_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
                self.set_edge_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,16,17] ) )
                self.set_edge_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) )
                self.set_edge_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
                self.set_edge_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) * ( bike_class != 2 ) )
                self.set_edge_attribute_value( (a,b), 'bike_vol',0)

        # add new dual (link-to-link) attribute columns
        self.add_dual_attribute('thru_centroid') # from centroid connector to centroid connector
        self.add_dual_attribute('l_turn') # left turn
        self.add_dual_attribute('u_turn') # u turn
        self.add_dual_attribute('r_turn') # right turn
        self.add_dual_attribute('turn') # turn
        self.add_dual_attribute('thru_intersec') # through a highway intersection
        self.add_dual_attribute('thru_junction') # through a junction

        self.add_dual_attribute('path_onoff') # movement in between bike path and other type

        self.add_dual_attribute('walk_cost') # total walk generalized cost
        self.add_dual_attribute('bike_cost') # total bike generalized cost

        # loop over pairs of edges and set attribute values
        for edge1 in self.dual:
            for edge2 in self.dual[edge1]:

                traversal_type = self.traversal_type(edge1,edge2,'auto_permit')

                self.set_dual_attribute_value(edge1,edge2,'thru_centroid', 1 * (traversal_type == 0) )
                self.set_dual_attribute_value(edge1,edge2,'u_turn', 1 * (traversal_type == 3 ) )
                self.set_dual_attribute_value(edge1,edge2,'l_turn', 1 * (traversal_type in [5,7,10,13]) )
                self.set_dual_attribute_value(edge1,edge2,'r_turn', 1 * (traversal_type in [4,6,9,11]) )
                self.set_dual_attribute_value(edge1,edge2,'turn', 1 * (traversal_type in [3,4,5,6,7,9,10,11,13]) )
                self.set_dual_attribute_value(edge1,edge2,'thru_intersec', 1 * (traversal_type in [8,12]) )
                self.set_dual_attribute_value(edge1,edge2,'thru_junction', 1 * (traversal_type == 14) )

                path1 = ( self.get_edge_attribute_value(edge1,'bike_class') == 1 )
                path2 = ( self.get_edge_attribute_value(edge2,'bike_class') == 1 )

                self.set_dual_attribute_value(edge1,edge2,'path_onoff', 1 * ( (path1 + path2) == 1 ) )

                if coef_walk:
                    self.set_dual_attribute_value(edge1,edge2,'walk_cost',self.calculate_variable_cost(edge1,edge2,coef_walk,0.0) )

                if coef_bike:
                    self.set_dual_attribute_value(edge1,edge2,'bike_cost',self.calculate_variable_cost(edge1,edge2,coef_bike,0.0) )
