Multivariate Bernoulli Mixture Model Sampler
============================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4122299.svg)](https://doi.org/10.5281/zenodo.4122299)

This project implements a Monte Carlo sampler for the above model.
When fed data on pair-contacts, it will automatically
identify prototypical shapes.

All the source files in this distribution were written by David M. Rogers.
If you make use of this code, we ask you to read and cite the following
reference:

David M. Rogers, Protein Conformational States: A First Principles Bayesian Method.
*submitted*, 2020. [arXiv:2008.02353](https://arxiv.org/abs/2008.02353).

Lots of helper programs are included to analyze protein conformations.

They are being made available under the terms of the GNU GPL version 3.
A copy of that license must accompany this work and all
derivatives (see `LICENSE.txt`).

Incuded Programs
----------------

This source distribution contains several top-level programs.

* `classify.py features.npy` runs the classification algorithm based on an `NxM` boolean matrix.
  The matrix should be stored in compressed format using numpy's
  [packbits](https://numpy.org/doc/stable/reference/generated/numpy.packbits.html).

  At present, the number of MCMC chains and samples to collect is selected by
  editing the source code.  Running this program generates `indices.npy`, listing the
  indices of important features used for classification.  It also generates
  several sub-directories named `sz1`, `sz2`, etc.  Each contins `features.npy`
  which is a `Kxlen(indices)` matrix of feature probabilities (0 <= mu <= 1).


* `reclassify.py indices.npy features.npy` assigns categories to each input sample,
  and saves a file, `rep.txt`, listing the population in each cluster and the
  1-based index of its "most representative" sample from the input.

* `pull.py` is a helper script that parses the `rep.txt` files from above
  and pulls out the corresponding frames from a DCD trajectory.
  This generates a multi-model PDB file showing the examplary
  structures from each conformational state.

* `plot.py` is a helper script that reads the category labels
  and permutes an all-all RMSD matrix - so that categories appear in blocks.

* `test_bbm.py` is the top-level testing routine that checks key properties
  of the implementation.  When run with the right command-line arguments,
  it reproduces the test results shown in the publication.

  Some helpful code bits for working with these test results are in
  the `test/` subdirectory.

* `compare2.py` calculates the [Bhattacharyya Similarity](https://en.wikipedia.org/wiki/Bhattacharyya_distance)
  between two different classifications.

* `diff.py` compares structures within each 

* `plotP.py` creates a scatter-plot of cluster assignment probabilities.
  The plot is projected down to 2D by using a weighted sum of vertices
  -- where the probability of assignment to each cluster is the weight.
  Each vertex is the corner of a K-sided polyhedron.

The programs above are mostly self-documenting.  A few of them
depend on `ucgrad`, which is available in one of the subdirectories
of [forcesolve](https://github.com/frobnitzem/forcesolve).
Email the author if you're interested in turning that into a formal python package.

Protein Helpers
---------------

The `prot_helpers` subdirectory contains useful source code for generating
pair distance maps.  The main program, `contacts.py`, passes each frame
to a user-specified python function which generates the frame's binary fingerprint.

An example fingerprint generator is provided as `mpro.py`.  This script
generates pair contacts that focus on a catalytic
site involving residues 41, 145, and 188 of chain A.


How to Contribute
-----------------

Submit an [issue](https://github.com/frobnitzem/classifier/issues) describing
your use case, and we'll discuss.  Some things on my wishlist include:

* maximization of the likelihood function from a given starting point

* A more user-friendly interface for `classify.py`

* Additional tests created from simple classification problems.

