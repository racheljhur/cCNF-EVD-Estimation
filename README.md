This repository contains research code associated with *Flow-Based Models for Estimating Exceedance Distributions in Unidirectional Polymer Matrix Composites*. The included datasets are (1) compressed microstructural descriptors for each microstructure as pc_scores.npy and (2) corresponding high values of maximum principal stress as responses.npy.

Because we're working with rves, you'll need to group the shuffled datapoints by their microstructural statistics. This is done using rve_assignments.py after running train.py. A much simpler alternative is to randomly permute indices and write this shuffled index set with the label for re-assignment post-training.

contact - jhur64@gatech.edu
