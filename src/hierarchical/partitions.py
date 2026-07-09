import numpy as np

class PartitionCell:
    # 1. FIX: Added bbox=None to the parameters
    def __init__(self, cell_id, point_indices, depth=0, bbox=None):
        self.id = cell_id
        self.point_indices = point_indices  
        self.parent = None
        self.children = []
        self.depth = depth                  # Root = 0
        self.generation = -1                # Leaves = 0
        self.bbox = bbox                    # tuple: (min_bounds, max_bounds)
    
    def __repr__(self):
        return f"Cell(id={self.id}, gen={self.generation}, pts={len(self.point_indices)})"


class HierarchicalPartition:
    def __init__(self, points, target_g=None, max_points_per_cell=8, max_allowed_depth=10):
        self.points = np.array(points, dtype=float)
        self.dimensions = self.points.shape[1] 
        self.cells = []
        self.target_g = target_g
        
        self.max_points_per_cell = max_points_per_cell
        self.max_allowed_depth = max_allowed_depth
        
        self._cell_counter = 0
        self._build_tree()

    def _build_tree(self):
        root_indices = list(range(len(self.points)))
        root_bbox = (np.min(self.points, axis=0), np.max(self.points, axis=0))
        
        # 2. FIX: Pass root_bbox to the root cell
        root_cell = PartitionCell(self._cell_counter, root_indices, depth=0, bbox=root_bbox)
        self.cells.append(root_cell)
        self._cell_counter += 1
        
        max_depth = self._split_recursive(root_cell, root_bbox)
        
        if self.target_g is not None:
            max_depth = max(max_depth, self.target_g - 1)
        
        self._pad_leaves(max_depth)
        
        self.g = max_depth + 1
        self.generations = [[] for _ in range(self.g)]
        
        for cell in self.cells:
            cell.generation = max_depth - cell.depth 
            self.generations[cell.generation].append(cell)

    def _split_recursive(self, current_cell, bbox):
        pts = self.points[current_cell.point_indices]
        depth = current_cell.depth
        
        if (len(pts) <= self.max_points_per_cell or 
            depth >= self.max_allowed_depth or 
            np.all(np.max(pts, axis=0) - np.min(pts, axis=0) < 1e-9)):
            return depth
            
        min_b, max_b = bbox
        mid_b = (min_b + max_b) / 2.0
        
        # map points to 2^d sub-quadrants using bitwise logic
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
                    
            # 3. FIX: Pass the newly calculated (new_min, new_max) to the child cell
            child_cell = PartitionCell(self._cell_counter, indices, depth=depth + 1, bbox=(new_min, new_max))
            self._cell_counter += 1
            child_cell.parent = current_cell
            current_cell.children.append(child_cell)
            self.cells.append(child_cell)
            
            d = self._split_recursive(child_cell, (new_min, new_max))
            max_child_depth = max(max_child_depth, d)
            
        return max_child_depth

    def _pad_leaves(self, max_depth):
        leaves = [c for c in self.cells if not c.children]
        for leaf in leaves:
            current = leaf
            while current.depth < max_depth:
                # 4. FIX: A padded leaf represents the exact same spatial box as its parent
                child = PartitionCell(self._cell_counter, current.point_indices, depth=current.depth + 1, bbox=current.bbox)
                self._cell_counter += 1
                child.parent = current
                current.children.append(child)
                self.cells.append(child)
                current = child
