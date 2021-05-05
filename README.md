# 2D Trajectory Solver

## Overview

This code implements Deep Reinforcement Learning as a technique for solving
2D Transfer Orbits. Motion is modeled by a 4th Order Runga-Kutta.
TF-Agents is used to implement the RL components.

The code is all contained within the `/code/` subdirectory. More specifically:
`/code/rksolvers.py` contains all code implementing the Runga-Kutta solver,
`/code/solar.py` implements classes for modeling the solar system and space missions,
`/code/rlenvs.py` formalizes tensorflow environments (the language of reinforcement
learning) using the solar classes, `/code/train.py` is a runnable script for training
and saving models, and finally `/code/test.py` is another runnable script which
tests fully-trained models to see how well they work.

The `/models/` subdirectory contains fully trained models that have been saved
for later use and `/plots/` contains images saved from training. `/resources/`
contains the presentation materials for the project.

This was created as my final project for PHYS 416 at Rice University in Spring 2021.

## Installation

This project requires a bunch of libraries outside the scope of this class.
The easiest way to manage the installs if through a virtual environment.

After downloading the code, use the following commands to make it run:

1. Navigate to the root directory in terminal
1. Run `python3 -m venv env` to create a virtual environment
1. Run `source env/bin/activate` to activate the environment
1. Run `pip install -r requirements.txt` to install project dependencies
1. Modify and run the code as you wish
    * `cd code` followed by `python3 test.py` displays my results
1. Run `deactivate` to deactivate the environment

You only need to create the virtual and and download project dependencies once.
This means if you want to edit the code or return to it at a later time, you
can ignore steps 1 and 4 since the virtual env is already created and contains
the necessary dependencies.
