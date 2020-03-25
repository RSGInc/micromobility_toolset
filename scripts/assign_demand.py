import choice_set, network, config, output, csv, sqlite3, numpy, time, argparse

def get_skim_matrix(net,taz_nodes,varcoef,max_cost=None):

	max_taz = max( taz_nodes.keys() )
	skim_matrix = numpy.zeros((max_taz,max_taz))

	for i in range(max_taz):

		centroid = taz_nodes[i+1]
		costs = net.single_source_dijkstra(centroid,varcoef,max_cost=max_cost)[0]

		for j in range(max_taz):

			if taz_nodes[j+1] in costs:
				skim_matrix[i,j] = costs[taz_nodes[j+1]]

	return skim_matrix

def load_trip_matrix(net,trips,load_name,taz_nodes,varcoef,max_cost=None):

	max_taz = max( taz_nodes.keys() )

	for i in range(max_taz):

		centroid = taz_nodes[i+1]
		paths = net.single_source_dijkstra(centroid,varcoef,max_cost=max_cost)[1]

		for j in range(max_taz):

			target = taz_nodes[j+1]

			if target in paths and trips[i,j] > 0:
				for k in range(len(paths[target])-1):
					edge = (paths[target][k],paths[target][k+1])
					net.set_edge_attribute_value(edge,load_name,trips[i,j]+net.get_edge_attribute_value(edge,load_name))

def read_taz_from_sqlite(config):

	result = {}

	# open database cursor
	database_connection = sqlite3.connect(config.application_config.base_sqlite_file)
	database_connection.row_factory  = sqlite3.Row
	database_cursor = database_connection.cursor()

	# execute select of link table
	database_cursor.execute('select * from ' + config.application_config.taz_table_name)

	# loop over database records
	while True:

		# get next record
		row = database_cursor.fetchone()

		if row is None:
			# if no more records we're done
			break
		else:
			taz = row[list(row.keys()).index(config.application_config.taz_taz_column)]
			node = row[list(row.keys()).index(config.application_config.taz_node_column)]
			county = row[list(row.keys()).index(config.application_config.taz_county_column)]
			result[taz] =  {'node': node, 'county': county}

	return result


def read_matrix_from_sqlite(config,table_name,sqlite_file):

	# open database cursor
	database_connection = sqlite3.connect(sqlite_file)
	database_cursor = database_connection.cursor()

	# execute select of link table
	database_cursor.execute('select * from ' + table_name)

	rows= database_cursor.fetchall()

	if len(rows[0])>3:
		dim = (config.application_config.num_zones,config.application_config.num_zones,len(rows[0])-2)
	else:
		dim = (config.application_config.num_zones,config.application_config.num_zones)

	trip_matrix = numpy.zeros(dim)

	if len(rows[0])>3:
		for row in rows:
			trip_matrix[row[0]-1,row[1]-1,:] = row[2:]
	else:
		for row in rows:
			trip_matrix[row[0]-1,row[1]-1] = row[2]

	return trip_matrix

def add_derived_network_attributes(net):

	net.add_edge_attribute('d0')
	net.add_edge_attribute('d1')
	net.add_edge_attribute('d2')
	net.add_edge_attribute('d3')
	net.add_edge_attribute('dne1')
	net.add_edge_attribute('dne2')
	net.add_edge_attribute('dne3')
	net.add_edge_attribute('dw')
	net.add_edge_attribute('riseft')
	net.add_edge_attribute('auto_permit')
	net.add_edge_attribute('bike_exclude')
	net.add_edge_attribute('dloc')
	net.add_edge_attribute('dcol')
	net.add_edge_attribute('dart')
	net.add_edge_attribute('dne3loc')
	net.add_edge_attribute('dne2art')
	net.add_edge_attribute('bike_vol')

	for a in net.adjacency:
		for b in net.adjacency[a]:
			distance = net.get_edge_attribute_value((a,b),'distance')
			bike_class = net.get_edge_attribute_value((a,b),'bike_class')
			lanes = net.get_edge_attribute_value((a,b),'lanes')
			# from_elev = net.get_edge_attribute_value((a,b),'from_elev')
			# to_elev = net.get_edge_attribute_value((a,b),'to_elev')
			link_type = net.get_edge_attribute_value((a,b),'link_type')
			fhwa_fc = net.get_edge_attribute_value((a,b),'fhwa_fc')
			net.set_edge_attribute_value( (a,b), 'd0', distance * ( bike_class == 0 and lanes > 0 ) )
			net.set_edge_attribute_value( (a,b), 'd1', distance * ( bike_class == 1 ) )
			net.set_edge_attribute_value( (a,b), 'd2', distance * ( bike_class == 2 ) )
			net.set_edge_attribute_value( (a,b), 'd3', distance * ( bike_class == 3 ) )
			net.set_edge_attribute_value( (a,b), 'dne1', distance * ( bike_class != 1 ) )
			net.set_edge_attribute_value( (a,b), 'dne2', distance * ( bike_class != 2 ) )
			net.set_edge_attribute_value( (a,b), 'dne3', distance * ( bike_class != 3 ) )
			net.set_edge_attribute_value( (a,b), 'dw', distance * ( bike_class == 0 and lanes == 0 ) )
			# net.set_edge_attribute_value( (a,b), 'riseft',  max(to_elev - from_elev,0) )
			net.set_edge_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
			net.set_edge_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
			net.set_edge_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
			net.set_edge_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,17] ) )
			net.set_edge_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,16,77] ) )
			net.set_edge_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
			net.set_edge_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,16,77] ) * ( bike_class != 2 ) )
			net.set_edge_attribute_value( (a,b), 'bike_vol',0)

	net.add_dual_attribute('thru_centroid')
	net.add_dual_attribute('l_turn')
	net.add_dual_attribute('u_turn')
	net.add_dual_attribute('r_turn')
	net.add_dual_attribute('turn')
	net.add_dual_attribute('thru_intersec')
	net.add_dual_attribute('thru_junction')

	net.add_dual_attribute('path_onoff')

	for edge1 in net.dual:
		for edge2 in net.dual[edge1]:

			traversal_type = net.traversal_type(edge1,edge2,'auto_permit')

			net.set_dual_attribute_value(edge1,edge2,'thru_centroid', 1 * (traversal_type == 0) )
			net.set_dual_attribute_value(edge1,edge2,'u_turn', 1 * (traversal_type == 3 ) )
			net.set_dual_attribute_value(edge1,edge2,'l_turn', 1 * (traversal_type in [5,7,10,13]) )
			net.set_dual_attribute_value(edge1,edge2,'r_turn', 1 * (traversal_type in [4,6,9,11]) )
			net.set_dual_attribute_value(edge1,edge2,'turn', 1 * (traversal_type in [3,4,5,6,7,9,10,11,13]) )
			net.set_dual_attribute_value(edge1,edge2,'thru_intersec', 1 * (traversal_type in [8,12]) )
			net.set_dual_attribute_value(edge1,edge2,'thru_junction', 1 * (traversal_type == 14) )

			path1 = ( net.get_edge_attribute_value(edge1,'bike_class') == 1 )
			path2 = ( net.get_edge_attribute_value(edge2,'bike_class') == 1 )

			net.set_dual_attribute_value(edge1,edge2,'path_onoff', 1 * ( (path1 + path2) == 1 ) )


if __name__ == '__main__':

	t1 = time.time()

	resources = config.Config()

	# parse command line options to get base and build database file locations
	parser = argparse.ArgumentParser(description='Perform incremental logit bike mode shift model')
	parser.add_argument('--type') #ignore here
	parser.add_argument('--base',dest='base',action='store')
	parser.add_argument('--build',dest='build',action='store')
	parser.add_argument('--base_disk',help='read base skims from disk to speed up incremental demand',action='store_true')
	args = parser.parse_args()
	resources.application_config.base_sqlite_file = args.base
	resources.application_config.build_sqlite_file = args.build

	resources.application_config.read_base_skims_from_disk = args.base_disk

	# store number of zones
	nzones = resources.application_config.num_zones

	base_net = network.Network(resources.network_config,resources.application_config.base_sqlite_file)

	add_derived_network_attributes(base_net)

	taz_data =  read_taz_from_sqlite(resources)

	taz_nodes ={}
	taz_county = {}
	for taz in taz_data:
		taz_nodes[taz] = taz_data[taz]['node']
		taz_county[taz] = taz_data[taz]['county']

	total_demand = numpy.zeros((nzones,nzones))

	print('getting demand matrices...')
	for idx in range(len(resources.mode_choice_config.trip_tables)):
		base_trips = read_matrix_from_sqlite(resources,resources.mode_choice_config.trip_tables[idx],resources.application_config.base_sqlite_file)
		bike_trips = base_trips[:,:,6]

		if resources.mode_choice_config.trip_tables[idx] != 'nhbtrip':
			bike_trips = 0.5 * (bike_trips + numpy.transpose(bike_trips))

		print('')
		print(('segment '+resources.mode_choice_config.trip_tables[idx]))
		print('non-intrazonal bike trips')
		print(int( numpy.sum( bike_trips * ( numpy.ones((nzones,nzones)) - numpy.diag(numpy.ones(nzones) ) ) ) ))

		total_demand = total_demand + bike_trips

	print('')
	print('assigning trips...')
	load_trip_matrix(base_net,total_demand,'bike_vol',taz_nodes,resources.mode_choice_config.route_varcoef_bike,resources.mode_choice_config.max_cost_bike)

	f = open('bike_vol.csv','wb')
	writer = csv.writer(f)

	print('writing results...')
	for a in base_net.adjacency:
		for b in base_net.adjacency[a]:
			writer.writerow([a,b,base_net.get_edge_attribute_value((a,b),'bike_vol')])

	f.close()
