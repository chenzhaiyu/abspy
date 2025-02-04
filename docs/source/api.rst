API Reference
===============

.. autoclass:: abspy.VertexGroup
    
    .. autofunction:: abspy.VertexGroup.__init__

    .. autofunction:: abspy.VertexGroup.load_file

    .. autofunction:: abspy.VertexGroup.process

    .. autofunction:: abspy.VertexGroup.get_points

    .. autofunction:: abspy.VertexGroup.get_primitives

    .. autofunction:: abspy.VertexGroup.normalise_from_centroid_and_scale

    .. autofunction:: abspy.VertexGroup.normalise_to_centroid_and_scale

    .. autofunction:: abspy.VertexGroup.fit_plane

    .. autofunction:: abspy.VertexGroup.save_vg

    .. autofunction:: abspy.VertexGroup.save_bvg

    .. autofunction:: abspy.VertexGroup.save_planes_txt

    .. autofunction:: abspy.VertexGroup.save_planes_npy

    .. autofunction:: abspy.VertexGroup.save_aabbs_npy

    .. autofunction:: abspy.VertexGroup.save_cloud


.. autoclass:: abspy.VertexGroupReference
    
    .. autofunction:: abspy.VertexGroupReference.__init__

    .. autofunction:: abspy.VertexGroupReference.process

    .. autofunction:: abspy.VertexGroupReference.perturb

    .. autofunction:: abspy.VertexGroupReference.inject_points

    .. autoproperty:: abspy.VertexGroupReference.bottom_indices

    .. autoproperty:: abspy.VertexGroupReference.wall_indices

    .. autofunction:: abspy.VertexGroupReference.save_vg

    .. autofunction:: abspy.VertexGroupReference.save_bvg

    .. autofunction:: abspy.VertexGroupReference.save_cloud


.. autoclass:: abspy.CellComplex
    
    .. autofunction:: abspy.CellComplex.__init__

    .. autofunction:: abspy.CellComplex.refine_planes

    .. autofunction:: abspy.CellComplex.prioritise_planes

    .. autofunction:: abspy.CellComplex.construct

    .. autofunction:: abspy.CellComplex.visualise

    .. autoproperty:: abspy.CellComplex.num_cells

    .. autoproperty:: abspy.CellComplex.num_planes

    .. autofunction:: abspy.CellComplex.volumes

    .. autofunction:: abspy.CellComplex.cell_representatives

    .. autofunction:: abspy.CellComplex.cells_in_mesh

    .. autofunction:: abspy.CellComplex.print_info

    .. autofunction:: abspy.CellComplex.save

    .. autofunction:: abspy.CellComplex.save_npy

    .. autofunction:: abspy.CellComplex.save_obj

    .. autofunction:: abspy.CellComplex.save_plm


.. autoclass:: abspy.AdjacencyGraph
    
    .. autofunction:: abspy.AdjacencyGraph.__init__

    .. autofunction:: abspy.AdjacencyGraph.load_graph

    .. autofunction:: abspy.AdjacencyGraph.assign_weights_to_n_links

    .. autofunction:: abspy.AdjacencyGraph.assign_weights_to_st_links
    
    .. autofunction:: abspy.AdjacencyGraph.cut

    .. autofunction:: abspy.AdjacencyGraph.save_surface_obj

    .. autofunction:: abspy.AdjacencyGraph.draw

    .. autofunction:: abspy.AdjacencyGraph.to_indices

    .. autofunction:: abspy.AdjacencyGraph.to_dict


.. autofunction:: abspy.attach_to_log
