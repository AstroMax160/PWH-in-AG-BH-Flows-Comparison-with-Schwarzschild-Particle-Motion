

This README explains the division of the folder "Codes" as seen by the public github accessible in the project repository appendix of the report.

As can be seen, it contains 2 subfolders and a MATLAB file

## FaradayAnalysis.m

This file is used to obtain the Faraday Threshold and the Faraday wavelength (resonance with droplet). It was used to generate Figure 4.1 in the report, but its parameters can be freely modified. This file also outputs in the MATLAB terminal the requirements for the horizon radius derived in section 4.1.3 of the report.

## PWH Final Versions

This contains the code for the standalone PWH simulations that were tested to work (as in, the simulation methods work). Parameters can be varied and different situations can be simulated. It contains three codes: 

- *PWH_CPU_u_version_Final*: Uses CPUs to solve, and uses the u method discussed in the Milewski paper. Faster than PDE versions, but not easily adaptable to add the backgorund flow. Used to find the walking parameters.

- *PWH_CPU_PDE_Final*: Solves the full PDE system without reducing the variables to u. Takes quite a lot longer, but it is more related to adding the background flow later. 

- *PWH_GPU_PDE_Final*: Same as above, but using GPUs and CPUs, which makes it much faster.

## Combined Model

This subfolder has 4 files, 2 related to the actual combined model, and 2 related to the General Relativity comparison. These are the main contributions from my project and are ready to be used. Used for all plots in sections 5 and 6 of my report.

**PWH AG model**
- *PWH_AG_GPU*: Main code for the project where all simulations for PWH with strictly radial background flow were run. 

- *PWH_AG_GPU_with_azimuthal*: Code for exploratory simulation with radial and rotational flow. At present not fully analysed in relation to the GR connection, so horizon definition is still kept from the Schwarzschild spacetime, but should be adapted to the Kerr spacetime. However, the actual droplet motion and background flow should already be correct for this type of background flow, just the horizon needs to be redefined. At present, B is manually set in the beginning of the code, and can be set to 0 to return the strictly background case.

**GR comparison**
- *PWH_GR_ParameterConversion*: Uses the conversion derived in section 6 of the report to relate initial PWH system conditions to GR. The inverse is also supported (PWH-AG <--> GR)

- *GR_Reference_Simulation*: Runs the GR trajectory simulation in nondimensional units to later compare with the walker simulation.

## NOTES ON THE OUTPUT

Since these codes were ran on a server through slurm, they do not produce immediate visuals for the output. The output of all simulation files are stored as .npz with coordinates, times, system parameters, etc. The specific name for each quantity saved can be seen at the very end of the respective codes. These .npz files and the codes used to visualise the output can be seen in the folder named "Results", more explanation on that in the mentioned folder. 

An additional note is that the output is very heavy if Save_frames is set to True. This is what stores the frames to later animate the evolution, and makes the file reach upwards of 15GB often. If set to False, all the data is still stored in the .npz file, except the frames. This still allows to perform trajectory and velocity analyses without needing a lot of space, since without frames they do not go over 9000kb.