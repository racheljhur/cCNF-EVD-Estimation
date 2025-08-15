# Flow-Based Models for Estimating EVDs in PMCs
This repository contains all datasets and codes relevant to the work summarized in Flow-Based Models for Estimating Extreme Value Distributions in Unidirectional Polymer Matrix Composites. 
The two datasets included are (1) low-rank approximations of microstructural descriptors for each microstructural realization and (2) corresponding extreme
values of maximum principal stress. 

In brief, the continuous normalizing flow estimates $p(\beta|\mathbf{\alpha}), \beta \in \\mathbb{R}, \mathbf{\alpha} \in \\mathbb{R}^{10}$, where $\beta$ is the extreme value parameter serving as a local
surrogate measure of the drivers of damage initation and $\alpha$ is a 10-dimensional vector of PC scores approximating the features of a microstructural volume.

>> Code usage order: train.py > inference.py > class_assignment.py > eval.py

Good luck!

Contact: jhur64@gatech.edu 
