# Alchemist-NN

A package for generating chemical structures using artificial
neural networks.

Large molecular structures are traditionally constructed by randomly
placing smaller structures into a simulation box.
This is labor-intensive and results in structures that are not
"ready" for simulation, display, or analysis without further
processing steps.

This package provides a framework for training and employing an AI
model to move atoms and molecules within the simulation box
in order to prepare large molecular structures.

Alchemist-NN contains command-line utilities:
`train` and `generate` to train and run the generation step.
It also provides several different layers that can be
connected in different ways to build the neural network model used.

Although the full methodology, trained weights, and results are not
yet final, this framework supports investigations into AI for generative
molecular structures.  It complements existing, published
Boltzmann generators, already known in the ML for chemistry
community, with a flexible, HPC-focused design.

## Installing

Included pyproject.toml file is used to run a regular pip install

    python3 -m venv venv
    source venv/bin/activate
    pip install .

For development, you can use

    pip install -e .[dev]

or uv,

    uv sync --extra dev

## Running

Use alchemist as a library,

    from alchemist.flows import LeapFrog

etc. See tests/ for examples.

QDπ dataset [paper](https://www.nature.com/articles/s41597-025-04972-3)
and download [Zeng, J., Giese, T., Goetz, A. & York, D. The QDπ dataset, training data for drug-like molecules and biopolymer fragments and their interactions https://doi.org/10.5281/zenodo.14970869 (2025).](https://doi.org/10.5281/zenodo.14970869)
