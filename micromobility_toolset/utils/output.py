import csv
import copy
import numpy

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
