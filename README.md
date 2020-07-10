# open-quantum-systems-project
Code written for simulation of open quantum systems, using the QuTiP package, for work done during my Bachelor's thesis. QuTiP is an open source package available at [here](http://qutip.org/). 

The first set of scripts in this repository were written to reproduce the results given in "Out-of-equilibrium open quantum systems: A comparison of approximate quantum master equation approaches with exact results", which can be found [here](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.062114) and [here](https://arxiv.org/abs/1511.03778v4). The second set of scripts were written to simulate a Heisenberg XXZ spin chain coupled to bosonic baths, and their results formed the bulk of my thesis.

## Tight-Binding Model (Set 1)
This model is the exact same model studied in the above paper. The entire system consists of a tight binding model of either fermions or bosons. N sites in the middle of the chain of sites are taken to be the system, while the rest of the sites (which may be finite or infinite) constitutes the bath. N is taken to be 2 in all our codes. The standard regimes required for the validity of master equations, such as low system bath coupling, and small reservoir correlation times are valid. We use 4 different methods to study our system, the local-Lindblad QME, the Redfield QME (or the correlation function evolution of the above paper), the Quantum Langevin method (for steady-state properties), and exact numerics. 

### Description of Code
All variables are the same as those defined in the [paper](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.062114) . 

**correlations_fermions** : Code simulates time evolution by computing the correlation function evolution using RK4, for fermions, and plots the occupation numbers as a function of time, for various values of g. This code is heavily commented, and is easy to understand.

**correlations_bosons** : Code simulates time evolution by computing the correlation function evolution using RK4, for bosons, and plots the occupation numbers as a function of time, for various values of g. Code can be easily modified to compute steadystates properties as a function of g. 

**exactnumerics_fermions** : Code simulates the time evolution of the entire system+bath setup, for fermions, by calculating evolution of correlation matrix of the entire system under unitary evolution. Theory is included in notes. 

**exactnumerics_bosons** : Code simulates the time evolution of the entire system+bath setup, for bosons, by calculating evolution of correlation matrix of the entire system under unitary evolution. Theory is included in notes. 

**redfield_bosons** : Code simulates the full Redfield equation evolution of the system, for bosons, using a truncated Hilbert space of each boson. Redfield evolution is computed using RK4.

**redfield_fig_2** : Code computes time evolution of the full Redfield equation, and reproduces fig2 of the paper. (Redfield portion only)

**redfield_fig_3** : Code computes time evolution of the full Redfield equation, and reproduces fig3 of the paper. (Redfield portion only)

**redfield_fig_4** : Code computes time evolution of the full Redfield equation, and reproduces fig4 of the paper. (Redfield portion only)

**redfield_ss** : Code computes directly, the steady state of the full redfield equation, using QuTiPs solvers. 

**langevin** : Code computes steady state correlation matrix from the quantum Langevin formalism. Refer to notes for the theory






##### Things to do : Make a Pdf doing all the derivations of this paper. Add it to the files section. 
