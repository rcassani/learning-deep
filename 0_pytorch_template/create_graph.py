#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates an SVG file with the graph model of a network
"""

import torch
import model_zoo
import hiddenlayer as hl

# Model
# 1. Model design and GPU capability
model_name = 'simple_mlp'
model = getattr(model_zoo, model_name)()

# Build HiddenLayer graph
im = hl.build_graph(model, torch.zeros([1, 784]), transforms=[])

# Save graph as SVG file
im.save('./'+ model_name + '.svg'  , 'svg')
