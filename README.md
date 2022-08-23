# DensePacking
### Reinforcement Learning Implementation

### Safe Restriction
* [Safe-Reinforcement-Learning-Baselines](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baselines)
* [Safe-Explorer](https://github.com/AgrawalAmey/safe-explorer)
* [Almost Surely Safe RL Using State Augmentation](https://github.com/huawei-noah/HEBO/tree/master/SAUTE)


### PPO Lagrange
* [PPO Lagrangian in Pytorch](https://github.com/akjayant/PPO_Lagrangian_PyTorch)

### Experiment Setup
<!--* remember to run the lab in the dpenvironment via the command `conda activate dp`-->
* `bash setup-v2.sh` each time on openning the gitpod ssh link.
* use the command `ssh -L 16006:127.0.0.1:6006 'kangoxford-densepacking-yky9rdbuiid@kangoxford-densepacking-yky9rdbuiid.ssh.ws-eu61.gitpod.io'` if you want to run in vscode
* use this link to get login via gitpod `https://kangoxford-densepacking-yky9rdbuiid.ws-eu61.gitpod.io`
* [Colab Files](https://drive.google.com/drive/folders/1SRJ1L5yqpOKNAy1KzWORoDuKI1PlbUSq?usp=sharing)
<!--* it should look like this `(dp) gitpod /workspace/DensePacking/DP_torch (main) $ `-->

# Action Space Designing
* Choice.01<br>
Uniform scaling (volume change) & deformation (shape change).<br>
![image](https://user-images.githubusercontent.com/37290277/185138157-6dd599a5-2a11-47c0-8140-1760e6e22382.png)


Then what is deformation?

![b8f1f6d68e00cfa97ec0090c2c8d64f8_hd](https://user-images.githubusercontent.com/72123149/185965879-a73886b4-9cf6-4862-a2d9-d9291f819e81.jpg)

* Choice.02<br>
Add a small strain of the fundamental cell, including both volume and shape changes.
![f0d9bdbd2124d551aab934f41b07e6f](https://user-images.githubusercontent.com/72123149/185965831-1c4ecf9f-59d6-4cfe-b860-b3459fe11953.png)

* Choice.03<br>
Here we employ random rotation on three vectors of the fundamental cell, and set their lengths as random variables.


# Basic definition
  A packing P is defined as a collection of non-overlapping (i.e., hard) objects or particles in either a finite-sized container or d-dimensional Euclidean space R^d. The packing density \fai is defined as the fraction of space R^d covered by the particles. A problem that has been a source of fascination to mathematicians and scientists for centuries is the determination of the densest arrangement(s) of particles that do not tile space and the associated maximal density \fai_max.


![figure1](https://user-images.githubusercontent.com/72123149/184534480-0f1a86f2-5d20-4975-8bed-7eb787dbc381.png)

# Dense packing
  Since the well-known Kepler’s conjecture in 1611 concerned with the densest sphere packing, it took nearly 400 years for this problem to be rigorously proved. This celebrated question was further highlighted by David Hilbert as the 18th problem, i.e., how one can arrange most densely in space an infinite number of equal solids of given form, hoping to guide mathematical research in the twentieth century. There have been many other attempts concerning optimal packings while remaining unsolved, of which we pay particular attention to Ulam’s conjecture stating that congruent spheres have the lowest optimal packing density of all convex bodies in R^3. Proving optimality in many 3D packing problems is surprisingly difficult.
  Comparatively much less is known about dense packings of hard non-spherical particle systems. Since non-spherical particles have both translational and rotational degrees of freedom, they usually have a richer phase diagram than spheres, i.e., the former can possess different degrees of translational and orientational order.
  
* It is important to introduce the lattice packing composed of nonoverlapping identical particles centered at the points of a lattice \Lambda with a common orientation, which, for an ordered packing, possibly corresponds to the maximally dense packings.

# Non-spherical particles
### Ellipsoid
* In 2004, Donev et al. [30] proposed a simple monoclinic crystal with two ellipsoids of different orientations per unit cell (SM2).
![figure2](https://user-images.githubusercontent.com/72123149/184534832-a22fdb2a-6d26-4572-acbf-9d685ac315bd.png)

* It was only recently that an unusual family of crystalline packings of biaxial ellipsoids was discovered, which are denser than the corresponding SM2 packings for a specific range of aspect ratios (like self-dual ellipsoids with 1.365<\alpha<1.5625.
![figure3](https://user-images.githubusercontent.com/72123149/184534880-ad3ba1bb-8cde-48ab-8ce0-6117c34490bd.png)
* Can denser packing been discovered via via a reinforcenment learning–based strategy?

# Methods
  To construct possible packing, we consider cases with repeating unit cell, which contains N parrtticles. The cell's lattice repetition is governed by the translation vectors, subject to the constraint that no two particle overlap.
![2009 Dense packings of polyhedra, Platonic and Archimedean solids](https://user-images.githubusercontent.com/72123149/184535539-f55f8d2a-f6ab-40bf-ae0a-25727a11426a.jpg)

* The number of particles N is small, typically N < 12.
* The three vectors that span the simulation cell are allowed to vary independently of each other in both their
length and orientation.
* We do not make any assumptions concerning the orientation of the box here. A commonly used choice for variable-box-shape simulations is to have one of the box vectors along the x axis, another vector in the positive part of
the xy-plane, and the third in the z > 0 half space.

# Gym environment
### Cell_gym
  We firstly concentrate on a subproblem in which particles are fixed (both centroid and orientation) with adaptive cell. 
* Objective function: the volume of cell (should be minized)
* Action space (12 variables): three sets of euler angle (for the rotation of cell vectors) + vector lengths
* Observation space: particle info (scaled coordinate + quaternion + aspherical shape) + three cell vectors
* (!!!) Reward function: The reduction in the volume of cell could possibly arisen from increasing overlap in the packing.

# Experiment.01

<img width="375" alt="image" src="https://user-images.githubusercontent.com/37290277/184924363-f6004a68-0cec-47ed-85e5-0105a24c5de9.png">

<img width="823" alt="image" src="https://user-images.githubusercontent.com/37290277/184924894-fb3d1d07-035c-4b02-8bc8-68bb75e36a0d.png">

