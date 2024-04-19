import numpy as np
from pulp import *

'''
Input:
points: pints: numpy array with [point1, point2, ...], where each 
point is [x,y,cluster_id]
PS: cluster_id not used

Output:
distances: numpy matrix with [[d11,d12,..], [d21,d22,...]], where dij is the 
distance beetween points i and j
'''
def compute_distances(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))  # Initialize the distances matrix

    for i in range(num_points):
        for j in range(num_points):
            # Calculate Euclidean distance between points i and j
            distances[i, j] = np.sqrt((points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2)

    return distances

'''
Input:
points: pints: numpy array with [point1, point2, ...], where each 
point is [x,y,cluster_id]
PS: cluster_id not used

Output:
xij: LP binary selection variables xij. 1 if point i is part 
of cluster j (cluster where center is j)
yj: LP binary selection variables yj. 1 if point j is a cluster
(cluster where center is j)
D: LP integer selection variables representing the maximum
distance between a point and the cluster (point at the center
of the cluster)
'''
def create_decison_variables(points):
    index_list = range(len(points))
    xij = LpVariable.dicts("X",(index_list,index_list),0,None,LpBinary)
    yj = LpVariable.dicts("y",(index_list),0,None,LpBinary)
    D = LpVariable("D",0,None,LpContinuous)
    return xij, yj, D

'''
Input:
points: pints: numpy array with [point1, point2, ...], where each 
point is [x,y,cluster_id]
PS: cluster_id not used
problem: pulp problem
xij: LP binary selection variables xij. 1 if point i is part 
of cluster j (cluster where center is j)
yj: LP binary selection variables yj. 1 if point j is a cluster
(cluster where center is j)
D: LP integer selection variables representing the maximum
distance between a point and the cluster (point at the center
of the cluster)
p: number of clusters
dij: numpy matrix with [[d11,d12,..], [d21,d22,...]], where dij is the 
distance beetween points i and j

Output:
problem: pulp problem, now updated with the restrictions
'''
def add_restrictions(points, problem, xij, yj, D, p, dij):
    points_list = range(len(points))

    for i in points_list:
        problem += lpSum([xij[i][j]*dij[i][j] for j in points_list]) <= D, "Maximum point {i} distance".format(i=i)
        
    for i in points_list:
        for j in points_list:
            problem += yj[j] >= xij[i][j],"Cluster {j} can have point {i}".format(j=j,i=i)

    problem += lpSum([yj[j] for j in points_list]) == p, "There are p clusters"
        
    for i in points_list:
        problem += lpSum([xij[i][j] for j in points_list]) == 1, "Point {i} is in a cluster".format(i=i)
        
    return problem

def get_ij_from_xij(variable_name):
    if((len(variable_name) != 5) or (variable_name[0] != 'X')):
        return None, None
    return int(variable_name[2]), int(variable_name[4])

def interpret_variables(points, variables):
    xij_matrix = np.zeros((len(points),len(points)))
    for variable in variables:
        variable_name, variable_value = variable.getName(), variable.varValue
        i,j = get_ij_from_xij(variable_name)
        if(i is not None and j is not None):
            xij_matrix[i][j] = float(variable_value) # G: possible problem here

    yj_line = np.zeros(len(points))
    for column_index in range(len(points)):
        if(np.sum(xij_matrix[:,column_index]) > 0):
            yj_line[column_index] = 1

    return xij_matrix, yj_line

def get_clusters_indexes_from_yj(yj_line):
    clusters_indexes_list = [] # len = p 
    for i in range(len(yj_line)):
        if(yj_line[i] == 1):
            clusters_indexes_list.append(i)
    return clusters_indexes_list

def set_points_cluster(points, xij_matrix):
    for point_index in range(len(points)):
        connected_cluster_index =get_clusters_indexes_from_yj(xij_matrix[point_index,:])
        assert(len(connected_cluster_index) == 1)
        points[point_index][2] = connected_cluster_index[0]
    return points

def solve_pcenter_pulp(points, p):
    distance_matrix = compute_distances(points)
    problem = LpProblem("p-center",LpMinimize)
    xij, yj, D = create_decison_variables(points)
    objective_function = D
    problem += objective_function
    problem = add_restrictions(points,problem,xij,yj,D,p,distance_matrix)
    
    problem.solve()
    print("Status: ", LpStatus[problem.status])
    for v in problem.variables():
        print(v.name, "=", v.varValue)
    
    xij_matrix, yj_line = interpret_variables(points, problem.variables())
    clusters_indexes_list = get_clusters_indexes_from_yj(yj_line)
    print("Clusters points: ", clusters_indexes_list)
    points = set_points_cluster(points, xij_matrix)
    return points, clusters_indexes_list

