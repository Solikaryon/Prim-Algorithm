#-*- coding: utf-8 -*-
# Prim's Algorithm - Minimum Spanning Tree on Image Graph
# @author: Monjaraz Briseno Luis Fernando

import cv2
import numpy as np
import math
import os

class PrimMSTVisualizer:
    """
    Prim's Algorithm implementation for finding the Minimum Spanning Tree
    on a graph represented by vertices on an image.
    """
    
    def __init__(self, map_name):
        """
        Initialize the Prim's Algorithm visualizer.
        
        Parameters:
        - map_name: Name/number of the map file (e.g., '3' for 'mapa3.png')
        """
        self.map_name = map_name
        self.map_image = None
        self.vertices = None
        self.valid_vertices = []
        self.mst_edges = []
        self.binary_image = None
        
        self._load_data()
    
    def _load_data(self):
        """Load map image and vertices from disk."""
        map_filename = f'mapa{self.map_name}.png'
        vertices_filename = f'verticeMapa{self.map_name}.npy'
        
        # Validate files exist
        if not os.path.exists(map_filename):
            raise FileNotFoundError(f'Error: Map file "{map_filename}" not found')
        if not os.path.exists(vertices_filename):
            raise FileNotFoundError(f'Error: Vertices file "{vertices_filename}" not found')
        
        # Load image and vertices
        self.map_image = cv2.imread(map_filename)
        if self.map_image is None:
            raise ValueError(f'Error: Failed to load map image "{map_filename}"')
        
        self.vertices = np.load(vertices_filename)
        print(f'Loaded map: {map_filename}')
        print(f'Loaded vertices: {vertices_filename} ({len(self.vertices)} vertices)')
    
    def _process_image(self):
        """Process image to identify valid vertex positions."""
        # Convert to grayscale
        gray = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to binary (finding white areas)
        _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        
        # Dilate to expand white areas
        kernel = np.ones((11, 11), np.uint8)
        binary = cv2.dilate(binary, kernel, 1)
        
        self.binary_image = binary
        
        # Convert back to BGR for drawing
        output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Identify valid vertices (those on white areas)
        for vertex in self.vertices:
            row, col = vertex[0], vertex[1]
            cv2.circle(output, (col, row), 3, (255, 0, 0), -1)
            
            # Check if vertex is on white area
            if binary[row, col] == 255:
                self.valid_vertices.append((col, row))
        
        print(f'Valid vertices found: {len(self.valid_vertices)}')
        
        return output
    
    def _has_obstacle(self, p1, p2):
        """
        Check if there's an obstacle (black pixel) between two points.
        Uses Bresenham's line algorithm for path checking.
        
        Parameters:
        - p1: First point (x, y)
        - p2: Second point (x, y)
        
        Returns:
        - True if obstacle found, False otherwise
        """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return False
        
        x_step, y_step = dx / steps, dy / steps
        
        for i in range(steps):
            x = int(x1 + i * x_step)
            y = int(y1 + i * y_step)
            
            # Check if pixel is black (obstacle)
            if self.binary_image[y, x] == 0:
                return True
        
        return False
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _prim_algorithm(self):
        """
        Implement Prim's algorithm to find the Minimum Spanning Tree.
        Starts from vertex 0 and greedily adds minimum weight edges.
        """
        num_vertices = len(self.valid_vertices)
        
        if num_vertices == 0:
            print('Error: No valid vertices found')
            return
        
        tree = [0]  # Start with first vertex
        self.mst_edges = []
        
        print('Running Prim\'s Algorithm...')
        
        while len(tree) < num_vertices:
            min_distance = float('inf')
            min_edge = None
            
            # For each vertex in the tree
            for v in tree:
                # For each vertex not in the tree
                for u in range(num_vertices):
                    if u not in tree:
                        p1 = self.valid_vertices[v]
                        p2 = self.valid_vertices[u]
                        
                        # Calculate distance
                        distance = self._calculate_distance(p1, p2)
                        
                        # Check if this is minimum and no obstacle
                        if distance < min_distance and not self._has_obstacle(p1, p2):
                            min_edge = (v, u)
                            min_distance = distance
            
            # Add minimum edge to tree
            if min_edge:
                tree.append(min_edge[1])
                self.mst_edges.append(min_edge)
        
        print(f'MST complete: {len(self.mst_edges)} edges added')
    
    def _draw_mst(self, image):
        """Draw the MST edges on the image."""
        for i, (v, u) in enumerate(self.mst_edges):
            p1 = self.valid_vertices[v]
            p2 = self.valid_vertices[u]
            cv2.line(image, p1, p2, (0, 255, 0), 2)
            
            # Calculate edge weight (distance)
            distance = self._calculate_distance(p1, p2)
            print(f'Edge {i}: Vertex {v} - Vertex {u} (distance: {distance:.2f})')
        
        return image
    
    def run(self):
        """Main execution method."""
        try:
            # Process the image
            result_image = self._process_image()
            
            # Find MST using Prim's algorithm
            self._prim_algorithm()
            
            # Draw the MST
            result_image = self._draw_mst(result_image)
            
            # Display result
            cv2.imshow(f'Prim MST - Map {self.map_name}', result_image)
            print('Display the result. Press any key to exit.')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f'Error: {e}')
            cv2.destroyAllWindows()


# Main execution
if __name__ == '__main__':
    map_number = '3'  # Change this to process different maps
    
    try:
        visualizer = PrimMSTVisualizer(map_number)
        visualizer.run()
    except FileNotFoundError as e:
        print(f'File error: {e}')
    except Exception as e:
        print(f'Unexpected error: {e}')