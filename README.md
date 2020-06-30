# open-quantum-systems-project
Code written for simulation of open quantum systems, using the QuTiP package, for work done during my Bachelor's thesis. QuTiP is an open source package available at [here](http://qutip.org/). 

The first set of scripts in this repository were written to reproduce the results given in "Out-of-equilibrium open quantum systems: A comparison of approximate quantum master equation approaches with exact results", which can be found [here](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.062114) and [here](https://arxiv.org/abs/1511.03778v4). The second set of scripts were written to simulate a Heisenberg XXZ spin chain coupled to bosonic baths, and their results formed the bulk of my thesis.

## Tight-Binding Model (Set 1)
This model is the exact same model studied in the above paper. The entire system consists of a tight binding model of either fermions or bosons. N sites in the middle of the chain of sites are taken to be the system, while the rest of the sites (which may be finite or infinite) constitutes the bath. N is taken to be 2 in all our codes. The standard regimes required for the validity of master equations, such as low system bath coupling, and small reservoir correlation times are valid. We use 4 different methods to study our system, the local-Lindblad QME, the Redfield QME (or the correlation function evolution of the above paper), the Quantum Langevin method (for steady-state properties), and exact numerics. 

### Description of Code


