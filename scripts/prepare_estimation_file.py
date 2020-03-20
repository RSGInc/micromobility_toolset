import choice_set, network, config, output, csv

def read_trip_data(filename):
	infile=open(filename,'rb')
	reader=csv.reader(infile)
	
	trip_data={}
	header={}
	count=0
	for row in reader:
		count=count+1
		if count==1:
			for i in range(len(row)):
				header[row[i]]=i
			continue
		if int(row[header['trip_id']]) not in trip_data:
			trip_data[int(row[header['trip_id']])]=[int(row[header['a']]),int(row[header['b']])]
		else:
			trip_data[int(row[header['trip_id']])].append(int(row[header['b']]))
			
	infile.close()

	return trip_data

if __name__ == '__main__':
	
	resources = config.Config()
	
	net = network.Network(resources.network_config)
	
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
	
	for a in net.adjacency:
		for b in net.adjacency[a]:
			distance = net.get_edge_attribute_value((a,b),'distance')
			bike_class = net.get_edge_attribute_value((a,b),'bike_class')
			lanes = net.get_edge_attribute_value((a,b),'lanes')
			from_elev = net.get_edge_attribute_value((a,b),'from_elev')
			to_elev = net.get_edge_attribute_value((a,b),'to_elev')
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
			net.set_edge_attribute_value( (a,b), 'riseft',  max(to_elev - from_elev,0) )
			net.set_edge_attribute_value( (a,b), 'bike_exclude', 1 * ( link_type in ['FREEWAY'] ) )
			net.set_edge_attribute_value( (a,b), 'auto_permit', 1 * ( link_type not in ['BIKE','PATH'] ) )
			net.set_edge_attribute_value( (a,b), 'dloc', distance * ( fhwa_fc in [19,9] ) )
			net.set_edge_attribute_value( (a,b), 'dcol', distance * ( fhwa_fc in [7,8,16,17] ) )
			net.set_edge_attribute_value( (a,b), 'dart', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) )
			net.set_edge_attribute_value( (a,b), 'dne3loc', distance * ( fhwa_fc in [19,9] ) * ( bike_class != 3 ) )
			net.set_edge_attribute_value( (a,b), 'dne2art', distance * ( fhwa_fc in [1,2,6,11,12,14,77] ) * ( bike_class != 2 ) )
			
	net.add_dual_attribute('thru_centroid')
	net.add_dual_attribute('l_turn')
	net.add_dual_attribute('u_turn')
	net.add_dual_attribute('r_turn')
	net.add_dual_attribute('turn')
	net.add_dual_attribute('thru_intersec')
	net.add_dual_attribute('thru_junction')
	
	net.add_dual_attribute('path_onoff')
	
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
			
	trip_data = read_trip_data(resources.trip_data_filename)
	
	for_deletion = []
	for trip_id in trip_data:
		for node in trip_data[trip_id]:
			if node not in net.adjacency:
				print ('Trip ' + str(trip_id) + ' excluded, node ' + str(node) + ' not in network')
				for_deletion.append(trip_id)
				break
	
	for trip_id in for_deletion:
		del trip_data[trip_id]
	
	paths = trip_data.values()
	
	"""
	bounding_box = choice_set.get_coef_bounding_box_noniterative(net,paths,resources.choice_set_config)
	print('Final bounding box:')
	print(bounding_box)
	
	bounding_box = {
		'distance':	[1.0,1.0],
		'dne1':		[0.082,6.41],
		'dne2':		[0.075,5.29],
		'dne3':		[0.098,10.8],
		'riseft':		[0.0009,0.0472],
		'dw':			[0.031,0.49],
		'turn':		[0.009,1.63],
		'thru_centroid': [999.0, 999.0], 
		'bike_exclude': [999.0, 999.0]
	}
	"""

	bounding_box = {
		'distance': [1.0, 1.0],
		'path_onoff': [0.04779991048771943, 0.058738602698877714],
		'thru_centroid': [998.9999999999711, 998.9999999999711],
		'turn': [0.1502848049719109, 7.683957036809541],
		'riseft': [0.005623413251903542, 0.5855975862829391],
		'bike_exclude': [998.9999999999711, 998.9999999999711],
		'dw': [0.044524623977534775, 0.42206542255192664],
		'dne1': [0.019897394524265467, 1.1950986283823033],
		'dne3': [0.028698188446724292, 5.7443613229667445],
		'dne2': [0.05489153702976999, 0.8951944140387113]
	}
	
	print('Generating choice sets...')
	choice_sets = {}
	for trip_id in trip_data:
		choice_sets[trip_id] = choice_set.generate_choice_set(net,trip_data[trip_id],bounding_box,resources.choice_set_config.randomization_scale,resources.choice_set_config.num_draws)
	
	print('Writing output files...')
	output.create_csv_from_choice_sets(choice_sets,resources.output_config.choice_set_pathname,resources.output_config.choice_set_linkname)
	output.create_estimation_dataset(net,choice_sets,resources.output_config)