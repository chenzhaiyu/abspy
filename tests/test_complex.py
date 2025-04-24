import pytest
import numpy as np
from pathlib import Path
from abspy.complex import CellComplex

@pytest.fixture
def simple_planes():
    """Create planes for testing using a verified working configuration."""
    # start from two planes
    planes = np.array([[0, 1, 0, -50], [0, 0, 1, -50]])

    # specify the bounds
    aabbs = np.array([[[0, 50, 0], [100, 50, 100]], [[0, 0, 50], [100, 50, 50]]])
    
    # Create arbitrary OBBs for testing
    obbs = np.array([
        [[0, 50, 0], [100, 50, 0], [100, 50, 100], [0, 50, 100]],   # First plane
        [[0, 0, 50], [100, 0, 50], [100, 50, 50], [0, 50, 50]]      # Second plane
    ])
    
    return planes, aabbs, obbs


@pytest.fixture
def simple_complex(simple_planes):
    """Create a simple cell complex."""
    planes, aabbs, obbs = simple_planes
    # Create points for each plane (simple grid of points)
    points = np.array([
        np.array([[x, 50, z] for x in np.linspace(0, 100, 5) for z in np.linspace(0, 100, 5)]),
        np.array([[x, y, 50] for x in np.linspace(0, 100, 5) for y in np.linspace(0, 50, 5)])
    ], dtype=object)
    
    # specify the initial bound
    initial_bound = np.array([[0, 0, 0], [100, 100, 100]])
        
    return CellComplex(planes, aabbs, obbs=obbs, points=points, 
                      initial_bound=initial_bound, build_graph=True, quiet=True)


def test_init(simple_planes):
    """Test CellComplex initialization."""
    planes, aabbs, obbs = simple_planes
    # Basic initialization
    cc = CellComplex(planes, aabbs, quiet=True)
    assert cc.planes.shape == planes.shape
    assert cc.aabbs.shape == aabbs.shape
    assert cc.obbs is None
    assert cc.graph is None
    assert cc.constructed is False
    assert len(cc.cells) == 1  # Initial cell
    
    # Initialization with graph building
    cc = CellComplex(planes, aabbs, build_graph=True, quiet=True)
    assert cc.graph is not None
    assert len(cc.graph.nodes) == 1  # Initial node
    
    # Initialization with obbs
    cc = CellComplex(planes, aabbs, obbs=obbs, quiet=True)
    assert cc.obbs.shape == obbs.shape


def test_construct(simple_complex):
    """Test cell complex construction."""
    cc = simple_complex
    cc.construct()
    
    # After construction, we should have more than the initial cell
    assert cc.constructed is True
    assert cc.num_cells == 4
    assert cc.num_planes == 2


def test_refine_planes(simple_complex):
    """Test plane refinement functionality."""
    cc = simple_complex
    
    # Save original plane data
    original_planes = cc.planes.copy()
    original_aabbs = cc.aabbs.copy()
    
    # Refine planes with default parameters
    cc.refine_planes()
    
    # Since our test data doesn't contain similar planes,
    # the number of planes should remain the same
    assert cc.planes.shape == original_planes.shape
    assert cc.aabbs.shape == original_aabbs.shape


def test_prioritise_planes(simple_complex):
    """Test plane prioritization functionality."""
    cc = simple_complex
    
    # Store original planes
    original_planes = cc.planes.copy()
    
    # Prioritize planes (without verticals first)
    cc.prioritise_planes(prioritise_verticals=False)
    
    # The total number should be the same
    assert cc.planes.shape == original_planes.shape
    assert (cc.planes == original_planes).all()  # no reorder
    
    # Test with vertical prioritization
    cc = simple_complex
    cc.prioritise_planes(prioritise_verticals=True)
    assert cc.planes.shape == original_planes.shape
    assert (cc.planes == original_planes).all()  # no reorder


def test_sort_planes():
    """Test plane sorting functionality."""
    # Create planes with different sized bounds
    planes = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ])
    
    # Use different sized AABBs
    aabbs = np.array([
        [[-1, -1, -0.01], [1, 1, 0.01]],     # Small z-extent
        [[-1, -0.01, -5], [1, 0.01, 5]],     # Large z-extent
        [[-0.01, -3, -3], [0.01, 3, 3]],     # Medium extent
    ])
    
    cc = CellComplex(planes, aabbs, quiet=True)
    
    # Test sorting by norm
    sorted_indices_norm = cc._sort_planes(mode='norm')
    assert sorted_indices_norm is not None
    assert len(sorted_indices_norm) == 3
    
    # Test sorting by volume
    sorted_indices_volume = cc._sort_planes(mode='volume')
    assert sorted_indices_volume is not None
    assert len(sorted_indices_volume) == 3
    
    # The sorted orders might differ between methods
    # Largest volume should be first
    volumes = np.prod(aabbs[:, 1, :] - aabbs[:, 0, :], axis=1)
    assert sorted_indices_volume[0] == np.argmax(volumes)


def test_inequalities():
    """Test the inequalities conversion function."""
    plane = np.array([1, 2, 3, -4])  # Plane equation: 1x + 2y + 3z - 4 = 0
    
    positive, negative = CellComplex._inequalities(plane)
    
    # Check format and values
    assert len(positive) == 4
    assert positive[0] == -4  # -4 + 1x + 2y + 3z >= 0
    assert positive[1:4] == [1, 2, 3]
    
    assert len(negative) == 4
    assert negative[0] == 4   # 4 - 1x - 2y - 3z >= 0
    assert negative[1:4] == [-1, -2, -3]


def test_cell_norm():
    """Test cell normal calculation."""
    # Create a simple rectangular set of vertices
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    # Calculate normals
    normals = CellComplex._cell_norm(vertices)
    
    # Check shape of results
    assert normals.shape == (3, 3)
    
    # For a cube, normals should be aligned with axes
    # We don't know the exact order, but they should be orthogonal
    for i in range(3):
        for j in range(i+1, 3):
            assert abs(np.dot(normals[i], normals[j])) < 1e-10


def test_visualise(simple_complex, tmpdir):
    """Test visualization functionality (mocking the actual display)."""
    # Monkey-patch trimesh.load_mesh to avoid actual rendering
    import trimesh
    original_load_mesh = trimesh.load_mesh
    
    try:
        # Create a mock for trimesh.load_mesh
        mock_mesh = type('MockMesh', (), {'show': lambda: None})
        trimesh.load_mesh = lambda *args, **kwargs: mock_mesh
        
        # Setup
        cc = simple_complex
        cc.construct()
        
        # Test visualization of all cells
        cc.visualise(temp_dir=str(tmpdir) + '/')
        
        # Test visualization of specific cells
        cc.visualise(indices_cells=[0, 1], temp_dir=str(tmpdir) + '/')
        
    finally:
        # Restore original function
        trimesh.load_mesh = original_load_mesh


def test_volumes(simple_complex):
    """Test volume calculation."""
    cc = simple_complex
    cc.construct()
    
    # Calculate volumes
    volumes = cc.volumes()
    
    # Check that all volumes are positive
    assert all(v > 0 for v in volumes)
    
    # Check with different engine
    volumes_sage = cc.volumes(engine='Sage')
    assert len(volumes) == len(volumes_sage)


def test_cell_representatives(simple_complex):
    """Test cell representative calculation."""
    cc = simple_complex
    cc.construct()
    
    # Test different representative types
    centers = cc.cell_representatives(location='center')
    centroids = cc.cell_representatives(location='centroid')
    
    # Check dimensions
    assert len(centers) == cc.num_cells
    assert len(centroids) == cc.num_cells
    
    # Test random representatives
    random_points = cc.cell_representatives(location='random_r', num=5)
    assert len(random_points) == cc.num_cells
    assert len(random_points[0]) == 5  # 5 points per cell
    
    # Test skeleton representatives 
    skeleton_points = cc.cell_representatives(location='skeleton', num=3)
    assert len(skeleton_points) == cc.num_cells


def test_cells_boundary(simple_complex):
    """Test boundary cell detection."""
    cc = simple_complex
    cc.construct()
    
    # Get boundary cells
    boundary_cells = cc.cells_boundary()
    
    # There should be some boundary cells
    assert len(boundary_cells) > 0
    # Verify they are integers
    assert all(isinstance(idx, (int, np.integer)) for idx in boundary_cells)


def test_save_load(simple_complex, tmpdir):
    """Test saving and loading."""
    cc = simple_complex
    cc.construct()
    
    # Save to temporary file
    filepath = Path(tmpdir) / "test_complex.cc"
    cc.save(filepath)
    
    # Check if file exists
    assert filepath.exists()
    
    # Test obj saving
    obj_path = Path(tmpdir) / "test_complex.obj"
    cc.save_obj(obj_path)
    assert obj_path.exists()
    
    # Test npy saving
    npy_path = Path(tmpdir) / "test_complex.npy"
    cc.save_npy(npy_path)
    assert npy_path.exists()


def test_invalid_construction():
    """Test error handling for invalid construction."""
    # Create a complex with minimal valid data
    planes = np.array([[0, 0, 1, 0]])  # single plane z=0
    aabbs = np.array([[[-1, -1, -0.01], [1, 1, 0.01]]])  # minimal AABB
    
    cc = CellComplex(planes, aabbs, quiet=True)
    assert cc.constructed is False
    
    # When trying to save an unconstructed complex, expect RuntimeError
    with pytest.raises(RuntimeError, match='cell complex has not been constructed'):
        cc.save("invalid.cc")
    
    with pytest.raises(RuntimeError, match='cell complex has not been constructed'):
        cc.save_obj("invalid.obj")


def test_pad_bound():
    """Test bound padding functionality."""
    bound = np.array([[-1, -1, -1], [1, 1, 1]])
    padding = 0.1
    
    # Create a cell complex
    cc = CellComplex(np.array([[0, 0, 1, 0]]), 
                    np.array([[[-1, -1, -0.1], [1, 1, 0.1]]]), 
                    quiet=True)
    
    # Test padding - the method is a static method
    padded_bound = CellComplex._pad_bound(bound, padding)
    
    # Check dimensions - make sure we're comparing scalars
    assert len(padded_bound) == 2
    assert padded_bound[0].shape == (3,)
    assert padded_bound[1].shape == (3,)
    
    # Calculate expected values
    extent = bound[1] - bound[0]
    expected_min = bound[0] - extent * padding
    expected_max = bound[1] + extent * padding
    
    # Use np.all to ensure we get a single boolean result
    assert np.all(np.isclose(padded_bound[0], expected_min))
    assert np.all(np.isclose(padded_bound[1], expected_max))


def test_normalize():
    """Test the normalize function."""
    # Create vectors with different magnitudes
    vectors = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])
    
    # Use the static normalize method directly
    normalized = CellComplex._normalize(vectors)
    
    # Check shapes - compare scalar values
    assert normalized.shape == vectors.shape
    
    # Check norms are 1 (with tolerance)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.all(np.isclose(norms, 1.0, atol=1e-10))
    
    # Check that very small vectors don't cause divide-by-zero
    tiny_vectors = np.array([[1e-15, 0, 0]])
    tiny_normalized = CellComplex._normalize(tiny_vectors)
    # Should have clipped the denominator to epsilon
    assert not np.any(np.isnan(tiny_normalized))
    assert not np.any(np.isinf(tiny_normalized))
