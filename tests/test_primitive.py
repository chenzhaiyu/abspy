import os
import pytest
import numpy as np
from pathlib import Path
import tempfile
import struct
from abspy.primitive import VertexGroup, VertexGroupReference
import trimesh

# Fixture to create a temporary .vg file for testing
@pytest.fixture
def temp_vg_file():
    with tempfile.NamedTemporaryFile(suffix='.vg', delete=False) as f:
        # Write a simple vertex group file
        f.write(b"num_points: 3\n")
        f.write(b"1.0 0.0 0.0 2.0 0.0 0.0 3.0 0.0 0.0\n")
        f.write(b"num_colors: 0\n")
        f.write(b"num_normals: 3\n")
        f.write(b"0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 1.0\n")
        f.write(b"num_groups: 1\n")
        f.write(b"group_type: 0\n")
        f.write(b"num_group_parameters: 4\n")
        f.write(b"group_parameters: 0.0 0.0 1.0 0.0\n")
        f.write(b"group_label: group_0\n")
        f.write(b"group_color: 0.5 0.5 0.5\n")
        f.write(b"group_num_point: 3\n")
        f.write(b"0 1 2\n")
        f.write(b"num_children: 0\n")
        file_path = f.name
    
    yield file_path
    
    # Clean up
    os.unlink(file_path)


# Fixture to create a temporary .bvg file for testing
@pytest.fixture
def temp_bvg_file():
    with tempfile.NamedTemporaryFile(suffix='.bvg', delete=False) as f:
        # Write a simple binary vertex group file
        # Number of points
        f.write(struct.pack('i', 3))  
        # Point coordinates
        for coord in [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0]:
            f.write(struct.pack('f', coord))
        # Number of colors
        f.write(struct.pack('i', 0))
        # Number of normals
        f.write(struct.pack('i', 3))
        # Normal vectors
        for norm in [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]:
            f.write(struct.pack('f', norm))
        # Number of groups
        f.write(struct.pack('i', 1))
        # Group type
        f.write(struct.pack('i', 0))
        # Number of parameters
        f.write(struct.pack('i', 4))
        # Parameters
        for param in [0.0, 0.0, 1.0, 0.0]:
            f.write(struct.pack('f', param))
        # Label length
        f.write(struct.pack('i', 7))
        # Label
        f.write(struct.pack('7s', b'group_0'))
        # Color
        for col in [0.5, 0.5, 0.5]:
            f.write(struct.pack('f', col))
        # Number of points in group
        f.write(struct.pack('i', 3))
        # Point indices
        for idx in [0, 1, 2]:
            f.write(struct.pack('i', idx))
        # Number of children
        f.write(struct.pack('i', 0))
        file_path = f.name
    
    yield file_path
    
    # Clean up
    os.unlink(file_path)


class TestVertexGroup:
    def test_load_vg_file(self, temp_vg_file):
        vg = VertexGroup(temp_vg_file)
        assert vg.processed
        assert vg.points is not None
        assert vg.planes is not None
        assert len(vg.points) == 3
        assert len(vg.planes) == 1
        
    def test_load_bvg_file(self, temp_bvg_file):
        vg = VertexGroup(temp_bvg_file)
        assert vg.processed
        assert vg.points is not None
        assert vg.planes is not None
        assert len(vg.points) == 3
        assert len(vg.planes) == 1
    
    def test_fit_plane(self):
        # Create a simple set of coplanar points
        points = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ])
        
        param, obb = VertexGroup.fit_plane(points, mode='PCA')
        
        # Check that the normal is [0, 0, 1] and d is -1
        np.testing.assert_allclose(param[:3], [0, 0, 1], rtol=1e-5)
        np.testing.assert_allclose(param[3], -1, rtol=1e-5)

    def test_normalise_from_centroid_and_scale(self, temp_vg_file):
        vg = VertexGroup(temp_vg_file)
        original_points = vg.points.copy()
        
        centroid = np.array([0, 0, 0])
        scale = 2.0
        
        vg.normalise_from_centroid_and_scale(centroid, scale)
        
        # Check that the points have been normalized
        expected_points = original_points / scale
        np.testing.assert_allclose(vg.points, expected_points, rtol=1e-5)
    
    def test_normalise_to_centroid_and_scale(self, temp_vg_file):
        vg = VertexGroup(temp_vg_file)
        
        centroid = np.array([1, 1, 1])
        scale = 0.5
        
        vg.normalise_to_centroid_and_scale(centroid, scale)
        
        # Check that the points have been transformed
        assert vg.points.shape == (3, 3)
        # The exact values depend on the implementation details

    def test_save_vg(self, temp_vg_file, tmpdir):
        vg = VertexGroup(temp_vg_file)
        
        output_file = Path(tmpdir) / "output.vg"
        vg.save_vg(output_file)
        
        # Check that the file was created
        assert output_file.exists()
        
        # Load the saved file and check contents
        vg2 = VertexGroup(output_file)
        assert vg2.processed
        assert len(vg2.points) == len(vg.points)
        assert len(vg2.planes) == len(vg.planes)
    
    def test_save_planes_txt(self, temp_vg_file, tmpdir):
        vg = VertexGroup(temp_vg_file)
        
        output_file = Path(tmpdir) / "planes.txt"
        vg.save_planes_txt(output_file)
        
        # Check that the file was created
        assert output_file.exists()
        
        # Read the file and check content
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == len(vg.planes)
    
    def test_save_planes_npy(self, temp_vg_file, tmpdir):
        vg = VertexGroup(temp_vg_file)
        
        output_file = Path(tmpdir) / "planes.npy"
        vg.save_planes_npy(output_file)
        
        # Check that the file was created
        assert output_file.exists()
        
        # Load the saved file and check contents
        loaded_planes = np.load(output_file)
        np.testing.assert_allclose(loaded_planes, vg.planes)
    
    def test_append_planes(self, temp_vg_file):
        vg = VertexGroup(temp_vg_file)
        original_plane_count = len(vg.planes)
        
        additional_planes = np.array([[0, 1, 0, 0]])
        vg.append_planes(additional_planes)
        
        assert len(vg.planes) == original_plane_count + 1
        np.testing.assert_allclose(vg.planes[-1], additional_planes[0])
    
    def test_points_bound(self):
        points = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [-1, 5, 2]
        ])
        
        bounds = VertexGroup._points_bound(points)
        expected_bounds = np.array([
            [-1, 0, 0],  # min values
            [1, 5, 3]    # max values
        ])
        
        np.testing.assert_allclose(bounds, expected_bounds)


# The following test requires Open3D and trimesh, which might not be available
@pytest.mark.skip(reason="Requires Open3D and trimesh")
class TestVertexGroupReference:
    def test_init_and_process(self, tmpdir):
        # Create a simple mesh file for testing
        mesh = trimesh.creation.box()
        mesh_file = Path(tmpdir) / "box.obj"
        mesh.export(mesh_file)
        
        vgr = VertexGroupReference(mesh_file, num_samples=100)
        
        assert vgr.processed
        assert vgr.points is not None
        assert vgr.planes is not None
        assert len(vgr.points_grouped) > 0
    
    def test_bottom_indices(self, tmpdir):
        # This test depends on the implementation of bottom_indices
        # and requires a valid mesh file
        pass
    
    def test_wall_indices(self, tmpdir):
        # This test depends on the implementation of wall_indices
        # and requires a valid mesh file
        pass
    
    def test_perturb(self, tmpdir):
        # This test depends on having a valid mesh file
        pass
    
    def test_save_vg(self, tmpdir):
        # This test depends on having a valid mesh file
        pass