from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import json
import yaml
import warnings
import copy
import os
from six.moves import zip

#from . import saving
from keras.engine.base_layer import Layer
from keras.engine.base_layer import Node
from keras.engine.input_layer import InputLayer
from keras import backend as K
from keras.utils.generic_utils import to_list

try:
    import h5py
except ImportError:
    h5py = None

# Note: This is an abridgement of a protected method from the Network class
# TBD: Review to ensure I haven't over imported

def map_network(inputs, outputs):
    """Gathers network's layers and nodes.

    # Arguments
        inputs: List of input tensors.
        outputs: List of outputs tensors.

    # Returns
         layers_by_depth: dict mapping ints (depth)
                          to lists of layer instances.

    # Raises
        ValueError: In case the network is not valid (e.g. disconnected graph).
    """

    # Normalize and set inputs, outputs.
    inputs = to_list(inputs, allow_tuple=True)
    outputs = to_list(outputs, allow_tuple=True)

    # Network_nodes: set of nodes included in the graph of layers
    # (not all nodes included in the layers are relevant to the current graph).
    nodes_depths = {}  # dict {node: depth value}
    layers_depths = {}  # dict {layer: depth value}
    layer_indices = {}  # dict {layer: index in traversal}
    nodes_in_decreasing_depth = []

    def build_map(tensor,
                  finished_nodes,
                  nodes_in_progress,
                  layer,
                  node_index,
                  tensor_index):
        """Builds a map of the graph of layers.

        This recursively updates the map `layer_indices`,
        the list `nodes_in_decreasing_depth` and the set `network_nodes`.

        # Arguments
            tensor: Some tensor in a graph.
            finished_nodes: Set of nodes whose subgraphs have been traversed
                completely. Useful to prevent duplicated work.
            nodes_in_progress: Set of nodes that are currently active on the
                recursion stack. Useful to detect cycles.
            layer: Layer from which `tensor` comes from. If not provided,
                will be obtained from `tensor._keras_history`.
            node_index: Node index from which `tensor` comes from.
            tensor_index: Tensor_index from which `tensor` comes from.

        # Raises
            ValueError: if a cycle is detected.
        """
        node = layer._inbound_nodes[node_index]

        # Prevent cycles.
        if node in nodes_in_progress:
            raise ValueError('The tensor ' + str(tensor) + ' at layer "' +
                             layer.name + '" is part of a cycle.')

        # Don't repeat work for shared subgraphs
        if node in finished_nodes:
            return

        # Store the traversal order for layer sorting.
        if layer not in layer_indices:
            layer_indices[layer] = len(layer_indices)

        nodes_in_progress.add(node)

        # Propagate to all previous tensors connected to this node.
        node_key = layer.name + '_ib-' + str(node_index)
        for i in range(len(node.inbound_layers)):
            x = node.input_tensors[i]
            layer = node.inbound_layers[i]
            node_index = node.node_indices[i]
            tensor_index = node.tensor_indices[i]
            build_map(x, finished_nodes, nodes_in_progress, layer,
                      node_index, tensor_index)

        finished_nodes.add(node)
        nodes_in_progress.remove(node)
        nodes_in_decreasing_depth.append(node)

    finished_nodes = set()
    nodes_in_progress = set()
    for x in outputs:
        layer, node_index, tensor_index = x._keras_history
        build_map(x, finished_nodes, nodes_in_progress,
                  layer=layer,
                  node_index=node_index,
                  tensor_index=tensor_index)
        
    for node in reversed(nodes_in_decreasing_depth):
        # If the depth is not set, the node has no outbound nodes (depth 0).
        depth = nodes_depths.setdefault(node, 0)

        # Update the depth of the corresponding layer
        previous_depth = layers_depths.get(node.outbound_layer, 0)
        # If we've seen this layer before at a higher depth,
        # we should use that depth instead of the node depth.
        # This is necessary for shared layers that have inputs at different
        # depth levels in the graph.
        depth = max(depth, previous_depth)
        layers_depths[node.outbound_layer] = depth
        nodes_depths[node] = depth

        # Update the depth of inbound nodes.
        # The "depth" of a node is the max of the depths
        # of all layers it is connected to.
        for i in range(len(node.inbound_layers)):
            inbound_layer = node.inbound_layers[i]
            node_index = node.node_indices[i]
            inbound_node = inbound_layer._inbound_nodes[node_index]
            previous_depth = nodes_depths.get(inbound_node, 0)
            nodes_depths[inbound_node] = max(depth + 1, previous_depth)

    # Build a dict {depth: list of layers with this depth}
    layers_by_depth = {}
    for layer, depth in layers_depths.items():
        if depth not in layers_by_depth:
            layers_by_depth[depth] = []
        layers_by_depth[depth].append(layer)

 
    return layers_by_depth
