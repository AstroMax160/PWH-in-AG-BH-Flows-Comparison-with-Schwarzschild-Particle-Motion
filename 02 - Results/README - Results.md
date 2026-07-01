
This README explains the division of the folder "Results" as seen by the public github accessible in the project repository appendix of the report.

This folder contains two subfolders each of which contains extra folders. The explanation detailed below will first explain certain things that both folders share, and at the end anything individual left is explained.

## Division

The two folders relate to saving aand visualising data, as well as performing analysis on the results. However, one folder contains the things from the PWH standalone simulation and the other has the things from the Combined model simulations, which contains more things due to the comparison with general relativity.

## Simulation Results

Both folders contain two types of results: .npz files, and .html files.

- *.npz files* contain all the metadata from the simulations be it for PWH standalone, the combined model, or the General Relativity simulations. They can be run with their respective running code (to be explained later) to see the animation, if the frames have been stored. Analysis can be done with these files.

- *.html files* have the analysed data already. For example, a plot of the droplet trajectory, or a file comparing the GR case with the PWH-AG case.

Below is a list of the results used in the report and to what simulation they are tied to. if html or npz is not specified, it is because the file exists in the two formats, and the beginning part of the name is the same for the html and npz versions.

**PWH Final Parameter Runs**: PWH_CPU_PDE_down_walking_speed (Used for figure 2.1 in the report)

**Combined Model / Anlogue Model Runs**: PWH_AG_CPU_v1_Stationary (Corresponds to *Simulation 1* in the report) - PWH_AG_GPU_Escape (*Simulation 2(a)*) - PWH_AG_GPU_Escape2 (*Simulation 2(b)*) - PWH_AG_GPU_EscapePhase (*Simulation 3(a)*) - PWH_AG_GPU_EscapePhase_SuperClose (*Simulation 3(b)*) - PWH_AG_GPU_CircOrbit (*Simulation 4*) - PWH_AG_GPU_CircOrbit_Azimuthal (*Exploratory Simulation*)

**Combined Model / Reference GR Model Runs**: Not directly shown in the report, bur needed for the comparison section

**Combined Model / GR and Analogue Full Analysis**: Stationary.html (Figure 6.1 of the report) - CircularOrbit_Base.html (Figures 6.4) 

# Visualising Results

AS explained in the codes section, the results are stored as .npz files. To obtain the htmls and animations, the two folders use special codes to extract the data and save it.

For PWH Final Parameter Runs, there are two relevant files:

- PlotWalkingSpeed_PWH.py makes an html based on the selected npz file where it shows the droplet total speed and does an averaging at the end to obtain the steady state walking speed.

- ShowResults_PWH.py takes the data from the .npz files and makes a figure appear. The figure shows the frames and some droplet information useful for debugging (such as height, contact and speed). This is the code to obtain stuff like Figure 5.1, but for the case without background flow. Do not run on the server if the server has no image display system. I have run this exclusively on my computer after downloading the data.

For the Combined Model folder, there are also 2 relevant files:

- ShowResults_AnalogueModel.py is equivalent to ShowResults_PWH.py, but with extra features, and meant for the case with the background flow. This file does not show the debugging info next to the animation, but rather all the velocity components (radial azimuthal and total). This file ALSO saves an html with all relevant data, such as the velocity plots and a trajectory plot. This file can be run even if the npz did not store frames, it will just not show any animation, but the html will still be created. Do not run on the server if the server has no image display system. I have run this exclusively on my computer after downloading the data.

- GR_&_Analogue_FullAnalysis.py takes the two chosen npzs related to the GR simulation and the PWH-AG simulation and puts them together to do the comparison. Shows trajectories separately, then the velocity components and differences between the GR and the analogue system, then the trajectory difference and then a plot with both trajectories overlaid. It also allows the user to change what quantities are used to nondimensionalise the system (in case, for example, the horizon is chosen to be different than 5cm which is what was used in the report).

