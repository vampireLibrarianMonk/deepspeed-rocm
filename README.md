# Foreword
This repo was created for anyone curious as to what is going on with AMD GPUs, ROCm (Radeon Open Compute) and the machine learning space. 

Please stop by the following repo first [amd-gpu-hello](https://github.com/vampireLibrarianMonk/amd-gpu-hello) to get the computer build and the environment setup.

# Getting Started
1. The following packages were installed to support plotting images for prediction verification.
```bash
pip3 install matplotlib mpi4py
```

[Matplotlib](https://pypi.org/project/matplotlib/) produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, Python/IPython shells, web application servers, and various graphical user interface toolkits.

[mpi4py](https://pypi.org/project/mpi4py/) provides Python bindings for the Message Passing Interface (MPI) standard. It is implemented on top of the MPI specification and exposes an API which grounds on the standard MPI-2 C++ bindings.

2. The following pypi packages were installed for cmath and to enable use of the pypi package mpi
```bash
sudo apt install g++-12 mpich -y
```

g++-12 refers to a specific version (12) of the g++ compiler, which is part of the GNU Compiler Collection (GCC). g++ is a compiler for C++ programming language. It is used to compile C++ source code into executable programs or libraries. The g++ compiler includes support for the C++ standard library, various optimizations, and debugging capabilities. Version 12 indicates a particular release of the g++ compiler, which would include updates, bug fixes, and possibly new features or enhancements over previous versions. This compiler is essential for developers writing applications in C++.

MPICH is a high-performance and widely portable implementation of the Message Passing Interface (MPI) standard. MPI is a standardized and portable message-passing system designed to function on parallel computing architectures. MPICH is used to enable communication in distributed computing environments, particularly in high-performance computing (HPC) systems. It is often used in scientific and engineering applications that require parallel processing and communication between different nodes in a computing cluster. MPICH allows these applications to efficiently perform computations across multiple processors in parallel, thereby enhancing performance for tasks that can be parallelized.

3. Pycharm was used to execute this project.  Ensure you use the following environment variables (6600 XT) when executing create_model, create_model_deepspeed and test_model.
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0
```