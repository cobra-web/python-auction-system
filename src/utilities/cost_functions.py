import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def squared_euclidean(X, Y):
    """
    Cost function for scenarios: P2H, P3H, P2I, grid.
    Computes the squared Euclidean distance between two sets of points.
    """
    # cdist with 'sqeuclidean' is highly optimized in C
    return cdist(X, Y, metric='sqeuclidean')

def euclidean(X, Y):
    """
    Cost function for scenario: P2H-P1.
    Computes the standard non-squared Euclidean distance.
    """
    return cdist(X, Y, metric='euclidean')

def compute_mesh_geodesic_matrix(vertices, faces, X_indices, Y_indices):
    """
    Cost function for scenario: mesh.
    Computes the shortest path (geodesic) along the edges of a 3D mesh graph.
    
    Parameters:
    - vertices: (V, 3) array of mesh vertex coordinates.
    - faces: (F, 3) array of triangle indices.
    - X_indices: The vertex indices where mass distribution mu_X resides.
    - Y_indices: The vertex indices where mass distribution mu_Y resides.
    """
    num_vertices = len(vertices)
    
    # 1. Build edges from faces
    edges = set()
    for face in faces:
        edges.add((min(face[0], face[1]), max(face[0], face[1])))
        edges.add((min(face[1], face[2]), max(face[1], face[2])))
        edges.add((min(face[2], face[0]), max(face[2], face[0])))
        
    edges = np.array(list(edges))
    v1, v2 = edges[:, 0], edges[:, 1]
    
    # 2. Compute Euclidean distance for each edge
    distances = np.linalg.norm(vertices[v1] - vertices[v2], axis=1)
    
    # 3. Create a sparse adjacency matrix
    adj_matrix = csr_matrix((np.concatenate([distances, distances]), 
                             (np.concatenate([v1, v2]), np.concatenate([v2, v1]))), 
                            shape=(num_vertices, num_vertices))
    
    # 4. Run Dijkstra's algorithm to find shortest paths from X points to all Y points
    # Return shape: (|X|, |Y|)
    geodesic_costs = dijkstra(csgraph=adj_matrix, directed=False, indices=X_indices)
    
    return geodesic_costs[:, Y_indices]

def bounding_box_lower_bound(boxA_min, boxA_max, boxB_min, boxB_max):
    """
    Cost function for scenario: P2H-LB.
    Computes the absolute minimum distance between two axis-aligned bounding boxes.
    Used in the Consistency Phase to avoid computing explicit pairwise min_c(x,y).
    """
    # The distance is 0 if the boxes overlap in a dimension, 
    # otherwise it is the gap between them.
    delta = np.maximum(0, np.maximum(boxA_min - boxB_max, boxB_min - boxA_max))
    
    # Return squared Euclidean distance of the gap
    return np.sum(delta ** 2)

def generate_grid_costs(grid_size_x, grid_size_y):
    """
    Helper for the 'grid' scenario.
    Generates a cost matrix for a smooth 2D mass distribution approximated by a discrete grid.
    """
    # Create x, y coordinates for the grid
    x = np.arange(grid_size_x)
    y = np.arange(grid_size_y)
    xv, yv = np.meshgrid(x, y)
    
    # Flatten to get a list of 2D coordinates
    points = np.vstack([xv.ravel(), yv.ravel()]).T
    
    # Grid scenario uses squared Euclidean distance
    return squared_euclidean(points, points)
