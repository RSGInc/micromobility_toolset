import choice_set, network, config, output, csv, sqlite3, numpy, time

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
			taz = row[row.keys().index(config.application_config.taz_taz_column)]
			node = row[row.keys().index(config.application_config.taz_node_column)]
			county = row[row.keys().index(config.application_config.taz_county_column)]
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