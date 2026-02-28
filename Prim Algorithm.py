#-*- coding: utf-8 -*-
# Prim's Algorithm - Minimum Spanning Tree on Image Graph (Enhanced Version)
# Created on Tuesday September 26 16:36:30 2023
# @author: Monjaraz Briseno Luis Fernando

import cv2
import numpy as np
import math
import os

class PrimMSTVisualizerEnhanced:
    """
    Enhanced version of Prim's Algorithm for finding the Minimum Spanning Tree
    with advanced image processing including morphological operations.
    """
    
    def __init__(self, map_name):
        """
        Initialize the enhanced Prim's Algorithm visualizer.
        
        Parameters:
        - map_name: Name/number of the map file (e.g., '3' for 'mapa3.png')
        """
        self.map_name = map_name
        self.original_map = None
        self.vertices = None
        self.valid_vertices = []
        self.mst_edges = []
        self.processed_image = None
        self.kernel = np.ones((11, 11), np.uint8)
        
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
        self.original_map = cv2.imread(map_filename)
        if self.original_map is None:
            raise ValueError(f'Error: Failed to load map image "{map_filename}"')
        
        self.vertices = np.load(vertices_filename)
        print(f'Loaded map: {map_filename}')
        print(f'Loaded vertices: {vertices_filename} ({len(self.vertices)} vertices)')
    
    def _process_image(self):
        """
        Advanced image processing with morphological operations.
        Includes threshold, dilation, erosion, and Gaussian blur for noise reduction.
        """
        print('Processing image...')
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_map, cv2.COLOR_BGR2GRAY)
        
        # Initial threshold (finding white areas)
        _, binary_1 = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        
        # Dilate to expand white areas
        binary_1 = cv2.dilate(binary_1, self.kernel, 1)
        
        # Erode to recover original boundaries
        binary_1 = cv2.erode(binary_1, self.kernel, 1)
        
        # Apply Gaussian blur for smoothing
        binary_1 = cv2.GaussianBlur(binary_1, (5, 5), cv2.BORDER_DEFAULT)
        
        # Second threshold after blur
        _, binary_2 = cv2.threshold(binary_1, 235, 255, cv2.THRESH_BINARY)
        
        # Final dilation for connectivity
        binary_2 = cv2.dilate(binary_2, self.kernel, 1)
        
        self.processed_image = binary_2
        
        # Convert to RGB for drawing
        output = cv2.cvtColor(binary_2, cv2.COLOR_GRAY2BGR)
        
        # Identify valid vertices (those on white areas)
        for vertex in self.vertices:
            row, col = vertex[0], vertex[1]
            cv2.circle(output, (col, row), 3, (255, 0, 0), -1)
            
            # Check if vertex is on white area (valid position)
            if binary_1[row, col] == 255:
                self.valid_vertices.append((col, row))
        
        print(f'Valid vertices found: {len(self.valid_vertices)} out of {len(self.vertices)}')
        
        return output
    
    def _has_obstacle(self, p1, p2):
        """
        Check if there's an obstacle (black pixel) between two points.
        Uses Bresenham's line algorithm for efficient path checking.
        
        Parameters:
        - p1: First point (x, y)
        - p2: Second point (x, y)
        
        Returns:
        - True if obstacle encountered, False otherwise
        """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return False
        
        x_step, y_step = dx / steps, dy / steps
        
        # Check each point along the line
        for i in range(steps):
            x = int(x1 + i * x_step)
            y = int(y1 + i * y_step)
            
            # Boundary check
            if y < 0 or y >= self.processed_image.shape[0] or \
               x < 0 or x >= self.processed_image.shape[1]:
                return True
            
            # Check if pixel is black (obstacle)
            if self.processed_image[y, x] == 0:
                return True
        
        return False
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _prim_algorithm(self):
        """
        Implement Prim's algorithm to find the Minimum Spanning Tree.
        
        Algorithm:
        1. Start with an arbitrary vertex
        2. Repeatedly add the minimum weight edge that connects
           a vertex in the tree to a vertex outside the tree
        3. Stop when all vertices are in the tree
        """
        num_vertices = len(self.valid_vertices)
        
        if num_vertices == 0:
            print('Error: No valid vertices found')
            return
        
        tree = [0]  # Start with first vertex
        self.mst_edges = []
        total_weight = 0.0
        
        print(f'Running Prim\'s Algorithm ({num_vertices} vertices)...')
        
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
                        
                        # Check if this is minimum distance and no obstacle
                        if distance < min_distance and not self._has_obstacle(p1, p2):
                            min_edge = (v, u)
                            min_distance = distance
            
            # Add minimum edge to tree
            if min_edge:
                tree.append(min_edge[1])
                self.mst_edges.append(min_edge)
                total_weight += min_distance
        
        print(f'MST complete: {len(self.mst_edges)} edges added')
        print(f'Total MST weight: {total_weight:.2f}')
    
    def _draw_mst(self, image):
        """
        Draw the MST edges on the image.
        
        Returns:
        - Image with MST edges drawn in green
        """
        print('Drawing MST edges...')
        
        for i, (v, u) in enumerate(self.mst_edges):
            p1 = self.valid_vertices[v]
            p2 = self.valid_vertices[u]
            
            # Draw edge
            cv2.line(image, p1, p2, (0, 255, 0), 2)
            
            # Calculate and print edge weight
            distance = self._calculate_distance(p1, p2)
            print(f'Edge {i}: Vertex {v} → Vertex {u} (distance: {distance:.2f})')
        
        return image
    
    def run(self):
        """Main execution method."""
        try:
            # Process the image with morphological operations
            result_image = self._process_image()
            
            # Find MST using Prim's algorithm
            self._prim_algorithm()
            
            # Draw the MST edges
            result_image = self._draw_mst(result_image)
            
            # Display result
            print('\n' + '='*50)
            print(f'Prim MST Visualization - Map {self.map_name}')
            print('='*50)
            print('Press any key to close the window.')
            
            cv2.imshow(f'Prim MST - Map {self.map_name}', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print('Program completed successfully.')
            
        except Exception as e:
            print(f'Error during execution: {e}')
            cv2.destroyAllWindows()


# Main execution
if __name__ == '__main__':
    # Configuration
    map_number = '3'  # Change this to process different maps (e.g., '1', '2', '3')
    
    try:
        print('=' * 50)
        print('Prim\'s Algorithm - MST Visualization (Enhanced)')
        print('=' * 50)
        
        visualizer = PrimMSTVisualizerEnhanced(map_number)
        visualizer.run()
        
    except FileNotFoundError as e:
        print(f'File Error: {e}')
        print('Make sure you have:')
        print('  - mapa3.png (map image)')
        print('  - verticeMapa3.npy (vertices data)')
        
    except Exception as e:
        print(f'Unexpected error: {e}')