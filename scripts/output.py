import csv, sqlite3, copy, numpy

def create_csv_from_path_list(path_list,pathname,linkname,dataNames=[],dataFrame=None):
	"""create two linked csv tables containing path links and path ids for viewing in GIS"""
	
	linkfile=open(linkname,'w')
	link_writer=csv.writer(linkfile,lineterminator='\r')
	
	pathfile=open(pathname,'w')
	path_writer=csv.writer(pathfile,lineterminator='\r')
	
	link_writer.writerow(['link_id','path_id','a','b'])
	path_writer.writerow(['path_id','orig','dest']+dataNames)
	
	for i in range(len(path_list)):
		curpath=path_list[i]
		if dataFrame is not None:
			temp=list(dataFrame[i])
		else:
			temp=[]
		path_writer.writerow([str(i),curpath[0],curpath[-1]]+temp)
		for j in range(len(curpath)-1):
			link_writer.writerow([str(j),str(i),curpath[j],curpath[j+1]])
	
	linkfile.close() 
	pathfile.close()
	
def create_csv_from_choice_sets(choice_sets,pathname,linkname):
	"""create two linked csv tables containing path links and path ids for viewing in GIS"""
	
	linkfile=open(linkname,'wb')
	link_writer=csv.writer(linkfile)
	
	pathfile=open(pathname,'wb')
	path_writer=csv.writer(pathfile)
	
	link_writer.writerow(['trip_id','path_id','link_id','a','b'])
	path_writer.writerow(['trip_id','path_id','orig','dest'])
	
	for trip_id in choice_sets:
		curset=choice_sets[trip_id]
		for i in range(len(curset)):
			curpath=curset[i]
			path_writer.writerow([trip_id,str(i),curpath[0],curpath[-1]])
			for j in range(len(curpath)-1):
				link_writer.writerow([trip_id,str(i),str(j),curpath[j],curpath[j+1]])
	
	linkfile.close() 
	pathfile.close()
	
def create_csv_from_matrix(matrix,filename):
	"""write matrix values to disk  in csv format, void"""
	
	f=open(filename,'wb')
	matrix_writer=csv.writer(f)
	
	matrix_writer.writerow(['i','j','value'])
	
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[0]):
			matrix_writer.writerow( [ i+1, j+1, matrix[i,j] ] )
	
	f.close() 

def create_estimation_dataset(net,choice_sets,output_config):
	
	est_file=open(output_config.estimation_file,'wb')
	est_writer=csv.writer(est_file)
	
	est_writer.writerow(['trip_id','alt','chosen']+output_config.variables+['path_size'])
	
	for trip_id in choice_sets:
		
		try:
			path_sizes = net.get_path_sizes(choice_sets[trip_id],output_config.path_size_overlap_var)
		except KeyError:
			print ('excluding trip '+str(trip_id)+', a node is missing in network')
			continue
		
		for alt_idx in range(len(choice_sets[trip_id])):
			path=choice_sets[trip_id][alt_idx]
			values=[]
			
			for i in range(len(output_config.variables)):
				key=output_config.variables[i]
				values.append(str(net.path_trace(path,key)))
			
			values.append(path_sizes[alt_idx])
			est_writer.writerow([str(trip_id),str(alt_idx),str(alt_idx==0)]+values)
	
	est_file.close()

def write_matrix_to_sqlite(matrix,sqlite_file,tablename,cores):
	
	# open database cursor
	database_connection = sqlite3.connect(sqlite_file)
	database_cursor = database_connection.cursor()
	
	columns = 'i integer, j integer'
	token = '(?,?'
	for c in cores:
		columns = columns + ', '+ c + ' real'
		token = token + ',?'
	token = token + ')'
	
	database_cursor.execute('create table if not exists ' + tablename + '(' + columns + ')' )
	
	database_cursor.close()
	database_connection.close()
	database_connection = sqlite3.connect(sqlite_file)
	database_cursor = database_connection.cursor()
	
	database_cursor.execute('delete from ' + tablename )

	if len(cores) > 1:
		for i in range(matrix.shape[0]):
			for j in range(matrix.shape[1]):
				if numpy.sum(matrix[i,j,:]) > 0:
					database_cursor.execute('insert into ' + tablename + ' values ' + token, tuple([i+1,j+1] +matrix[i,j,:].tolist()) )
	else:
		for i in range(matrix.shape[0]):
			for j in range(matrix.shape[1]):
				if matrix[i,j] > 0:
					database_cursor.execute('insert into ' + tablename + ' values ' + token, tuple([i+1,j+1] +[float(matrix[i,j])]) )
	
	database_connection.commit()