#!/usr/bin/bash

python generate_images_for_dataset.py

tar -czf geometric_shapes_images.tar.gz geometric_shapes_images
