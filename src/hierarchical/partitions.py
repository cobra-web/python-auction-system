import numpy as np

class PartitionCell:
    def __init__(self, cell_id, point_indices, depth=0, bbox=None):
        self.id = cell_id
        self.point_indices = point_indices  
        self.parent = None
        self.children = []
        self.depth = depth                  # Root = 0..
        self.bbox = bbox                    # tuple: (min_bounds, max_bounds)
    
    def __repr__(self):
        return f"Cell(id={self.id}, depth={self.depth}, pts={len(self.point_indices)})"

class HierarchicalPartition:
    def __init__(self, points, max_points_per_cell=8, max_allowed_depth=10):
        self.points = np.array(points, dtype=float)
        self.dimensions = self.points.shape[1] 
        self.cells = []
        
        self.max_points_per_cell = max_points_per_cell
        self.max_allowed_depth = max_allowed_depth
        
        self._cell_counter = 0
        self.max_depth = 0
        self._build_tree()

    def _build_tree(self):
        root_indices = list(range(len(self.points)))
        root_bbox = (np.min(self.points, axis=0), np.max(self.points, axis=0))
        
        root_cell = PartitionCell(self._cell_counter, root_indices, depth=0, bbox=root_bbox)
        self.cells.append(root_cell)
        self._cell_counter += 1
        
        self.max_depth = self._split_recursive(root_cell, root_bbox)

    def _split_recursive(self, current_cell, bbox):
        pts = self.points[current_cell.point_indices]
        depth = current_cell.depth
        
        if (len(pts) <= self.max_points_per_cell or 
            depth >= self.max_allowed_depth or 
            np.all(np.max(pts, axis=0) - np.min(pts, axis=0) < 1e-9)):
            return depth
            
        min_b, max_b = bbox
        mid_b = (min_b + max_b) / 2.0
        
        sub_indices = {i: [] for i in range(2**self.dimensions)}
        
        for idx in current_cell.point_indices:
            p = self.points[idx]
            quadrant = 0
            for dim in range(self.dimensions):
                if p[dim] >= mid_b[dim]:
                    quadrant |= (1 << dim)
            sub_indices[quadrant].append(idx)
            
        max_child_depth = depth
        
        for quad, indices in sub_indices.items():
            if not indices:
                continue 
                
            new_min = np.copy(min_b)
            new_max = np.copy(max_b)
            for dim in range(self.dimensions):
                if (quad & (1 << dim)):
                    new_min[dim] = mid_b[dim]
                else:
                    new_max[dim] = mid_b[dim]
                    
            child_cell = PartitionCell(self._cell_counter, indices, depth=depth + 1, bbox=(new_min, new_max))
            self._cell_counter += 1
            child_cell.parent = current_cell
            current_cell.children.append(child_cell)
            self.cells.append(child_cell)
            
            d = self._split_recursive(child_cell, (new_min, new_max))
            max_child_depth = max(max_child_depth, d)
            
        return max_child_depth

    def get_active_cells_at_depth(self, d):
        """Returns all cells perfectly tessellating the space at depth d without padding"""
        active = []
        for cell in self.cells:
            if cell.depth == d:
                active.append(cell)
            elif cell.depth < d and not cell.children:
                active.append(cell)
        return active
