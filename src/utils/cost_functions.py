import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def squared_euclidean(X, Y):
    #cdist with 'sqeuclidean' is highly optimized in C
    return cdist(X, Y, metric='sqeuclidean')

def euclidean(X, Y):
    #P2H-P1 aka standard non-squared Euclidean distance
    return cdist(X, Y, metric='euclidean')

def compute_mesh_geodesic_matrix(vertices, faces, X_indices, Y_indices):
    num_vertices = len(vertices)
    
    edges = set()
    for face in faces:
        edges.add((min(face[0], face[1]), max(face[0], face[1])))
        edges.add((min(face[1], face[2]), max(face[1], face[2])))
        edges.add((min(face[2], face[0]), max(face[2], face[0])))
        
    edges = np.array(list(edges))
    v1, v2 = edges[:, 0], edges[:, 1]
    
    #euclidean distance for each edge
    distances = np.linalg.norm(vertices[v1] - vertices[v2], axis=1)
    
    #sparse adjacency matrix
    adj_matrix = csr_matrix((np.concatenate([distances, distances]), 
                             (np.concatenate([v1, v2]), np.concatenate([v2, v1]))), 
                            shape=(num_vertices, num_vertices))
    
    #find shortest paths from X points to all Y points, return like (|X|, |Y|)
    geodesic_costs = dijkstra(csgraph=adj_matrix, directed=False, indices=X_indices)
    
    return geodesic_costs[:, Y_indices]

def bounding_box_lower_bound(boxA_min, boxA_max, boxB_min, boxB_max):
    #function for scenario: P2H-LB.
    #distance = 0 if the boxes overlap in a dimension, otherwise it's the gap between them.
    delta = np.maximum(0, np.maximum(boxA_min - boxB_max, boxB_min - boxA_max))
    
    #return squared Euclidean distance of the gap
    return np.sum(delta ** 2)

def generate_grid_costs(grid_size_x, grid_size_y):
    x = np.arange(grid_size_x)
    y = np.arange(grid_size_y)
    xv, yv = np.meshgrid(x, y)
    
    #flatten for a list of 2D coords
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    
    return squared_euclidean(points, points)
