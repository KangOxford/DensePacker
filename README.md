# DensePacking
### Reinforcement Learning Implementation

* remember to run the lab in the dpenvironment via the command `conda activate dp`
* or just `bash setup.sh` each time on openning the gitpod ssh link.
* it should look like this `(dp) gitpod /workspace/DensePacking/DP_torch (main) $ `


# Basic definition
  A packing P is defined as a collection of non-overlapping (i.e., hard) objects or particles in either a finite-sized container or d-dimensional Euclidean space R^d. The packing density \fai is defined as the fraction of space R^d covered by the particles. A problem that has been a source of fascination to mathematicians and scientists for centuries is the determination of the densest arrangement(s) of particles that do not tile space and the associated maximal density \fai_max.


![图片1](https://user-images.githubusercontent.com/72123149/184534480-0f1a86f2-5d20-4975-8bed-7eb787dbc381.png)

# Dense packing
  Since the well-known Kepler’s conjecture in 1611 concerned with the densest sphere packing, it took nearly 400 years for this problem to be rigorously proved. This celebrated question was further highlighted by David Hilbert as the 18th problem, i.e., how one can arrange most densely in space an infinite number of equal solids of given form, hoping to guide mathematical research in the twentieth century. There have been many other attempts concerning optimal packings while remaining unsolved, of which we pay particular attention to Ulam’s conjecture stating that congruent spheres have the lowest optimal packing density of all convex bodies in R^3. Proving optimality in many 3D packing problems is surprisingly difficult.
  Comparatively much less is known about dense packings of hard non-spherical particle systems. Since non-spherical particles have both translational and rotational degrees of freedom, they usually have a richer phase diagram than spheres, i.e., the former can possess different degrees of translational and orientational order.
  
* It is important to introduce the lattice packing composed of nonoverlapping identical particles centered at the points of a lattice \Lambda with a common orientation, which, for an ordered packing, possibly corresponds to the maximally dense packings.

# Non-spherical particles
# Ellipsoid
* In 2004, Donev et al. [30] proposed a simple monoclinic crystal with two ellipsoids of different orientations per unit cell (SM2).
![图片2](https://user-images.githubusercontent.com/72123149/184534832-a22fdb2a-6d26-4572-acbf-9d685ac315bd.png)

* It was only recently that an unusual family of crystalline packings of biaxial ellipsoids was discovered, which are denser than the corresponding SM2 packings for a specific range of aspect ratios (like self-dual ellipsoids with 1.365<\alpha<1.5625.
![图片3](https://user-images.githubusercontent.com/72123149/184534880-ad3ba1bb-8cde-48ab-8ce0-6117c34490bd.png)
* Can denser packing been discovered via via a reinforcenment learning–based strategy?

# Methods
  To construct possible packing, we consider cases with repeating unit cell, which contains N parrtticles. The cell's lattice repetition is governed by the translation vectors, subject to the constraint that no two particle overlap.
![2009 Dense packings of polyhedra, Platonic and Archimedean solids](https://user-images.githubusercontent.com/72123149/184535539-f55f8d2a-f6ab-40bf-ae0a-25727a11426a.jpg)

* The number of particles N is small, typically N < 12.
* The three vectors that span the simulation cell are allowed to vary independently of each other in both their
length and orientation.
* We do not make any assumptions concerning the orientation of the box here. A commonly used choice for variable-box-shape simulations is to have one of the box vectors along the x axis, another vector in the positive part of
the xy-plane, and the third in the z > 0 half space.

