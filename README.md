# CILNumericalCode
 This repository contains numerical code for the paper "Statistical approach for parameter identification by Turing patterns", published in the Journal of Theoretical Biology (see the link: https://www.sciencedirect.com/science/article/pii/S0022519320301740). The code is implemented in MATLAB programming language. For time-consuming parts we also provide optimized codes, written on C++ and CUDA. The scripts for the compilation of MEX files have been tested on MATLAB R2019b.
 
 ## UPDATE
  An updated version of the code, suitable for working with the multi-feature and synthetic-based CIL approaches, is located in the respective sub-folder of the repository. Additionally, new model examples, such as mechano-chemical and reaction-diffusion-ODE systems have been added. The new code can be used independently from the main part of the library. Sources for the MEX files are located in the 'MEX' sub-folder. 
 
 ## Contents
 There are three main scripts in the repository:
 1. "CIL_simu.m" This script creates the correlation integral likelihood (CIL) and estimates the mean and covariance of eCDF vectors.
 2. "CIL_de.m" This script estimates the model parameters by applying the Differential Evolution (DE) algorithm of stochastic optimization.
 3. "CIL_mcmc.m" This script constructs the posterior distribution of model parameters. PLEASE NOTE that it uses the MCMC toolbox for Matlab (MCMCSTAT) developed by Marko Laine. The toolbox is available on GitHub: https://github.com/mjlaine/mcmcstat Please download it before using the script.
 
 ## Compiling mex files
 We strongly recommend to build the mex files BEFORE using the code. Although this step is not mandadory for using the code, mex-files contain optimized implementations for time-consuming routines, which can significantly improve the computational time. We provide a fast C++ routine for distance matrix computation and both C++ and CUDA versions of numerical solvers for the reaction-diffusion systems, studied in the paper (FitzHugh-Nagumo model, Gierer-Meinhardt system and Brusselator reaction-diffusion system).  The source code for both C++ and CUDA implementations is available in MEX/Source subfolder. We provide two MATLAB scripts for compilation: "compile_all_cpp.m" and "compile_all_cuda.m". The first script builds the MEX files for C++ implementation of distance matrix computation and numerical solvers for reaction-diffusion systems. The second script builds the CUDA solvers for the respective models. If you choose to use CUDA solvers, please run the first script anyway to compile the distance matrix MEX file. Arter the successful compilation, MEX files are automatically recognized and the fastest available algorithm is used.
 
 ## Reaction-diffusion models
 There are three reaction-diffusion systems, included in the library: FitzHugh-Nagumo model, Gierer-Meinhardt system and Brusselator reaction-diffusion system. It is possible, however, to use other models with the same code as well. To do this, please derive a subclass from abstract class Model (located in Models subfolder).
 
 ## Versions
 All MATLAB code, included into the current repository, has been written and tested on MATLAB R2019b. For compiling MEX files we have used Visual C++ 2017 compiler and Nvidia CUDA Toolkit 10.1
 
 
