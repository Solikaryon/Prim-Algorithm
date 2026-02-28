# Prim's Algorithm - Minimum Spanning Tree Visualization

A Python implementation of Prim's algorithm for finding the Minimum Spanning Tree (MST) in a graph represented by vertices on an image. Includes advanced image processing and visualization capabilities.

## Description

This project implements **Prim's algorithm** to find the Minimum Spanning Tree of a weighted graph where:
- **Vertices** are specific points on an image
- **Edges** connect vertices with their weight being the Euclidean distance
- **Obstacles** are black pixels that block connections

The implementation comes in two versions:
1. **Prim Algorithm (Default).py** - Basic version with core functionality
2. **Prim Algorithm.py** - Enhanced version with advanced image processing

## Features

- **Prim's Algorithm Implementation** - Finds MST efficiently
- **Image-Based Graph** - Vertices loaded from image coordinates
- **Obstacle Detection** - Respects black pixels as obstacles using line scanning
- **Advanced Image Processing** - Morphological operations (dilation, erosion, Gaussian blur)
- **Real-Time Visualization** - Displays graph and MST edges on image
- **Class-Based Architecture** - Clean, reusable OOP design
- **Comprehensive Error Handling** - File validation and error messages
- **Multiple Implementations** - Choose between basic and enhanced versions

## Requirements

- Python 3.6 or higher
- OpenCV (cv2)
- NumPy

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install opencv-python numpy
```

## File Requirements

Your working directory must contain:
- `mapa{N}.png` - Image file defining the graph (N = map number)
- `verticeMapa{N}.npy` - NumPy file with vertex coordinates

**Example:**
- `mapa3.png` - Map image for map number 3
- `verticeMapa3.npy` - Vertices for map number 3

## Usage

### Using the Default Version (Basic)

```bash
python "Prim Algorithm (Deafault).py"
```

**Edit the script to change map number:**
```python
if __name__ == '__main__':
    map_number = '3'  # Change this value
    visualizer = PrimMSTVisualizer(map_number)
    visualizer.run()
```

### Using the Enhanced Version (Recommended)

```bash
python "Prim Algorithm.py"
```

**Edit the script to change map number:**
```python
if __name__ == '__main__':
    map_number = '3'  # Change this value
    visualizer = PrimMSTVisualizerEnhanced(map_number)
    visualizer.run()
```

### What Happens

1. Loads the map image and vertex coordinates
2. Processes the image to identify white walkable areas
3. Filters vertices to only those on white areas
4. Runs Prim's algorithm to find the MST
5. Displays the result with edges drawn in green
6. Prints edge information to console

## How It Works

### Prim's Algorithm Overview

Prim's algorithm finds the minimum spanning tree by:
1. Starting with an arbitrary vertex
2. Maintaining a set of vertices in the current tree
3. Repeatedly finding the minimum weight edge from the tree to a vertex outside
4. Adding that vertex and edge to the tree
5. Continuing until all vertices are included

### Algorithm Complexity

- **Time Complexity:** O(V²) with basic implementation, O((V + E) log V) with binary heap
- **Space Complexity:** O(V) for storing the tree

Where V = number of vertices, E = number of edges

### Pseudocode

```
Prim(G, w):
    T = {v₁}  // Start with arbitrary vertex
    E = {}     // Set of edges in tree
    
    while |T| < |V|:
        Find minimum weight edge (u, v):
            u ∈ T, v ∉ T
            
        Add v to T
        Add (u, v) to E
    
    return (T, E)
```

### Graph Representation in This Implementation

1. **Vertices** - Loaded from `.npy` file, filtered by image
2. **Edges** - Implicit (every pair of valid vertices is connected)
3. **Weights** - Euclidean distance between vertices
4. **Constraints** - Path must not cross black pixels (obstacles)

### Image Processing Steps

**Basic Version:**
```
Load Image
    ↓
Convert to Grayscale
    ↓
Threshold (find white areas)
    ↓
Dilate (expand white regions)
    ↓
Filter vertices (keep those on white)
    ↓
Run Prim's Algorithm
```

**Enhanced Version:**
```
Load Image
    ↓
Convert to Grayscale
    ↓
Threshold & Dilate
    ↓
Erode (recover boundaries)
    ↓
Gaussian Blur (smooth noise)
    ↓
Second Threshold
    ↓
Dilate (ensure connectivity)
    ↓
Filter vertices
    ↓
Run Prim's Algorithm
```

## Code Structure

### PrimMSTVisualizer Class (Basic Version)

Main class for basic Prim's algorithm:

```python
class PrimMSTVisualizer:
    def __init__(map_name)         # Initialize
    def _load_data()               # Load image and vertices
    def _process_image()           # Process image, filter vertices
    def _has_obstacle(p1, p2)      # Check if path has obstacles
    def _calculate_distance(p1, p2) # Compute Euclidean distance
    def _prim_algorithm()          # Run Prim's algorithm
    def _draw_mst(image)           # Draw MST edges on image
    def run()                      # Main execution
```

### PrimMSTVisualizerEnhanced Class (Enhanced Version)

Similar to basic version but with advanced image processing:
- Additional morphological operations
- Gaussian blur for noise reduction
- More detailed console output
- Boundary checking in obstacle detection

## Customization Guide

### Changing the Map

```python
# In main execution section:
map_number = '1'    # Load mapa1.png and verticeMapa1.npy
map_number = '2'    # Load mapa2.png and verticeMapa2.npy
map_number = '3'    # Load mapa3.png and verticeMapa3.npy
```

### Adjusting Image Processing Parameters

**Threshold values** (for detecting white areas):
```python
# Higher threshold = stricter color requirements
_, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
```

**Kernel size** for morphological operations:
```python
# Larger kernel = more aggressive processing
kernel = np.ones((11, 11), np.uint8)  # Change kernel size
```

**Gaussian blur kernel**:
```python
# Larger kernel = more blurring
binary = cv2.GaussianBlur(binary, (5, 5), cv2.BORDER_DEFAULT)
```

### Creating Custom Maps

To create your own map and vertex files:

```python
import cv2
import numpy as np

# 1. Create or prepare a map image
# - White areas for walkable paths
# - Black areas for obstacles
# Save as 'mapaN.png'
map_image = cv2.imread('mapaN.png')

# 2. Define vertex coordinates manually or automatically
vertices = np.array([
    [50, 100],    # [row, col]
    [150, 200],
    [200, 300],
    # ... more vertices
])

# 3. Save vertices
np.save('verticeMapaN.npy', vertices)
```

## Advanced Usage

### Debugging Obstacle Detection

Add visualization of path checking:

```python
def _has_obstacle_debug(self, p1, p2):
    # ... existing code ...
    
    # Visualize the path
    path_image = self.processed_image.copy()
    cv2.line(path_image, p1, p2, (128, 128, 128), 1)  # Draw path
    
    # Debug display (optional)
    if obstacle_found:
        cv2.imshow('Obstacle Path', path_image)
```

### Custom Vertex Filtering

Extend vertex validation with additional criteria:

```python
def _process_image_custom(self):
    # ... existing processing ...
    
    for vertex in self.vertices:
        row, col = vertex[0], vertex[1]
        
        # Basic white-area check
        if self.processed_image[row, col] != 255:
            continue
        
        # Additional filters:
        # - Distance from image border
        if row < 20 or row > image_height - 20:
            continue
        
        # - Distance from other vertices
        is_too_close = False
        for other in self.valid_vertices:
            dist = self._calculate_distance((col, row), other)
            if dist < 30:  # Minimum 30-pixel separation
                is_too_close = True
                break
        
        if not is_too_close:
            self.valid_vertices.append((col, row))
```

### Adding Edge Weights Visualization

Display edge weights on the image:

```python
def _draw_mst_with_weights(self, image):
    for i, (v, u) in enumerate(self.mst_edges):
        p1 = self.valid_vertices[v]
        p2 = self.valid_vertices[u]
        
        # Draw edge
        cv2.line(image, p1, p2, (0, 255, 0), 2)
        
        # Calculate weight
        distance = self._calculate_distance(p1, p2)
        
        # Draw weight label at midpoint
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(image, f'{distance:.0f}', (mid_x, mid_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
```

## Performance Optimization

### For Large Graphs (Many Vertices)

Use a priority queue-based implementation:

```python
import heapq

def _prim_algorithm_optimized(self):
    num_vertices = len(self.valid_vertices)
    tree = {0}
    edges = []
    
    # Add all edges from vertex 0
    for u in range(num_vertices):
        if u != 0:
            dist = self._calculate_distance(
                self.valid_vertices[0],
                self.valid_vertices[u]
            )
            heapq.heappush(edges, (dist, 0, u))
    
    while len(tree) < num_vertices and edges:
        dist, v, u = heapq.heappop(edges)
        
        if u in tree:
            continue
        
        tree.add(u)
        self.mst_edges.append((v, u))
        
        # Add new edges from u
        for w in range(num_vertices):
            if w not in tree:
                new_dist = self._calculate_distance(
                    self.valid_vertices[u],
                    self.valid_vertices[w]
                )
                heapq.heappush(edges, (new_dist, u, w))
```

### Reducing Image Processing Time

For large images, downscale before processing:

```python
# Downscale for processing, upscale for display
scale_factor = 0.5
small_map = cv2.resize(self.original_map, None, 
                      fx=scale_factor, fy=scale_factor)

# Process small image
# ... (process small_map instead of original)

# Upscale vertices back to original size
self.valid_vertices = [(int(x/scale_factor), int(y/scale_factor)) 
                       for x, y in self.valid_vertices]
```

## Algorithm Comparison

| Algorithm | Time | Space | Best For |
|-----------|------|-------|----------|
| Prim's | O(V²) | O(V) | Dense graphs, simple implementation |
| Kruskal's | O(E log E) | O(V) | Sparse graphs, parallel processing |
| Borůvka's | O(E log V) | O(V) | Dense graphs, theoretical interest |

## Applications

Prim's algorithm is used in:

1. **Network Design** - Minimizing cable length in network topology
2. **Circuit Design** - Routing with minimal wire length
3. **Image Processing** - Seam carving and graph-based segmentation
4. **Path Planning** - Finding minimum spanning trees for robotics
5. **Clustering** - Building minimum spanning tree clusters
6. **Approximation Algorithms** - TSP approximation

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Error: Map file not found" | Missing `mapaN.png` | Create or place map image in working directory |
| "Error: Vertices file not found" | Missing `verticeMapaN.npy` | Create or place vertices file in working directory |
| "No valid vertices found" | No white pixels in map | Adjust threshold or check map image |
| MST edges look wrong | Vertices filtered incorrectly | Adjust image processing parameters |
| Algorithm runs slowly | Too many vertices | Reduce map resolution or use optimized version |

## Algorithm Visualization Tips

1. **Verify white areas** - Ensure walkable paths are properly detected
2. **Check vertex distribution** - Confirm vertices are correctly placed
3. **Monitor edge drawing** - Verify that edges don't cross obstacles
4. **Print edge weights** - Debug by checking distance values

## Example Output

```
Loaded map: mapa3.png
Loaded vertices: verticeMapa3.npy (50 vertices)
Processing image...
Valid vertices found: 45 out of 50
Running Prim's Algorithm (45 vertices)...
MST complete: 44 edges added
Total MST weight: 8432.50

Edge 0: Vertex 0 → Vertex 5 (distance: 150.32)
Edge 1: Vertex 5 → Vertex 12 (distance: 98.45)
Edge 2: Vertex 12 → Vertex 8 (distance: 125.67)
...
```

## Dependencies Overview

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | >= 4.0 | Image processing and display |
| NumPy | >= 1.19 | Array operations, vertex management |
| Math | Built-in | Distance calculations |

## Educational Value

This implementation demonstrates:

1. **Graph Theory** - Understanding MST concepts
2. **Greedy Algorithms** - How Prim's greedy approach works
3. **Image Processing** - Morphological operations and thresholding
4. **Algorithm Visualization** - Converting abstract algorithms to visual output
5. **Python OOP** - Class design and encapsulation
6. **Computational Geometry** - Distance calculations and path checking

## Version Differences

| Feature | Default | Enhanced |
|---------|---------|----------|
| Morphological Operations | Dilate only | Dilate, Erode, Blur, Dilate |
| Noise Reduction | Minimal | Gaussian blur |
| Console Output | Basic | Detailed with progress |
| Boundary Checking | Basic | Advanced |
| Processing Time | Faster | Slightly slower |
| Result Quality | Good | Better for noisy images |

## Author

**Monjaraz Briseno Luis Fernando**

## License

This project is provided as-is for educational purposes.

## References

- Prim's Algorithm Wikipedia: https://en.wikipedia.org/wiki/Prim%27s_algorithm
- OpenCV Morphological Transformations: https://docs.opencv.org/4.x/d9/df8/tutorial_py_morphological_ops.html
- Introduction to Algorithms (CLRS): Chapter on Minimum Spanning Trees

## Future Enhancements

Possible improvements:

1. **Kruskal's Algorithm Implementation** - Compare with Prim's
2. **Performance Optimization** - Binary heap-based priority queue
3. **Interactive Visualization** - GUI for parameter adjustment
4. **Graph Export** - Save MST in standard formats (GML, GraphML)
5. **Parallel Processing** - Multi-threaded vertex scanning
6. **3D Visualization** - Extend to 3D graphs
7. **Dynamic Updates** - Real-time graph modifications

---

For questions, issues, or contributions, please contact the author.
