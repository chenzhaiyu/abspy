# File format specification for VertexGroup 

Planar primitives in a point cloud are defined in `VertexGroup` (a group of vertices for each primitive). The specification is as follows:

```c++
*  File format specification
         *      \code
         *       num_points: N   // N is an integer denoting the number of points
         *       x1  y1  z1	// 3 floating point numbers
         *       ...
         *       xN  yN  zN
         *
         *       // the colors of the points
         *       num_colors: N      // N is an integer denoting the number of colors (can be 0; if not, it must equal to num_points)
         *       r1 g1 b1	        // 3 floating point numbers
         *       ...
         *       rN gN bN
         *
         *       // the normals of the points
         *       num_normals: N     // N is an integer denoting the number of normals (can be 0; if not, it must equal to num_points)
         *       nx1  ny1  nz1	    // 3 floating point numbers
         *       ...
         *       nxN  nyN  nzN
         *
         *       // now we store the segmentation information
         *       num_groups: M      // M is an integer denoting the number of segments/primitives/objects in this point cloud (can be 0)
         *
         *       // now the information for the 1st segment/primitive/object
         *       group_type: type           // integer denoting the of the segment (0: PLANE, 1: CYLINDER, 2: SPHERE, 3: CONE, 4: TORUS, 5: GENERAL)
         *       num_group_parameters: NUM_GROUP_PARAMETERS    // integer number denoting the number of floating point values representing the segment (e.g., 4 for planes)
         *       group_parameters: float[NUM_GROUP_PARAMETERS] // a sequence of NUM_GROUP_PARAMETERS floating point numbers (e.g., a, b, c, and d for a plane)
         *       group_label: label         // the label (a string) of the segment
         *       group_color: r g b         // 3 floating point numbers denoting the color of this segment
         *       group_num_points: N        // N is an integer denoting the number of points in this segment (can be 0)
         *       id1 ... idN                // N integer numbers denoting the indices of the points in this segment
         *       num_children: num          // a segment/primitive/object may contain subsegment (that has the same representation as this segment)
         *       ...
         *       group_type: type           // integer denoting the of the segment (0: PLANE, 1: CYLINDER, 2: SPHERE, 3: CONE, 4: TORUS, 5: GENERAL)
         *       num_group_parameters: NUM_GROUP_PARAMETERS    // integer number denoting the number of floating point values representing the segment (e.g., 4 for planes)
         *       group_parameters: float[NUM_GROUP_PARAMETERS] // a sequence of NUM_GROUP_PARAMETERS floating point numbers (e.g., a, b, c, and d for a plane)
         *       group_label: label         // the label (a string) of the segment
         *       group_color: r g b         // 3 floating point numbers denoting the color of this segment
         *       group_num_points: N        // N is an integer denoting the number of points in this segment (can be 0)
         *       id1 ... idN                // N integer numbers denoting the indices of the points in this segment
         *       num_children: num          // a segment/primitive/object may contain subsegment (that has the same representation as this segment)
         *       ...
         *       \endcode
         */
```

To create such data from a point cloud, you can use `Mapple` in [Easy3D](https://github.com/LiangliangNan/Easy3D).

