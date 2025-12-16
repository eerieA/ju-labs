<!-- TOC -->

- [0. Preconditions (Windows side)](#0-preconditions-windows-side)
    - [0.1 Enable WSL 2 and install Ubuntu](#01-enable-wsl-2-and-install-ubuntu)
    - [0.2 Install NVIDIA Windows driver (critical)](#02-install-nvidia-windows-driver-critical)
- [1. Base system setup (inside WSL)](#1-base-system-setup-inside-wsl)
- [2. Install OpenMPI (system MPI)](#2-install-openmpi-system-mpi)
- [(Optional) 3. Install CUDA toolkit](#optional-3-install-cuda-toolkit)
    - [3.1 DO NOT install Windows CUDA toolkit](#31-do-not-install-windows-cuda-toolkit)
    - [3.2 Add NVIDIA CUDA repository (inside WSL)](#32-add-nvidia-cuda-repository-inside-wsl)
    - [3.3 Install CUDA toolkit](#33-install-cuda-toolkit)
- [4. Install Julia](#4-install-julia)
- [5. Julia packages: MPI, CUDA, IJulia](#5-julia-packages-mpi-cuda-ijulia)
    - [5.1 Configure MPI.jl to use system OpenMPI](#51-configure-mpijl-to-use-system-openmpi)
    - [5.2 Verify CUDA.jl](#52-verify-cudajl)
- [6. Install JupyterLab](#6-install-jupyterlab)
    - [6.1. Install pipx (system-managed, safe)](#61-install-pipx-system-managed-safe)
    - [6.2. Install JupyterLab via pipx](#62-install-jupyterlab-via-pipx)
    - [6.3. Register Julia with this Jupyter installation](#63-register-julia-with-this-jupyter-installation)
    - [6.4. Launch JupyterLab](#64-launch-jupyterlab)

<!-- /TOC -->

---

## 0. Preconditions (Windows side)

### 0.1 Enable WSL 2 and install Ubuntu

We already did this, but for completeness:

```powershell
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

### 0.2 Install NVIDIA Windows driver (critical)

Install a **WSL-compatible NVIDIA driver**:

* [https://developer.nvidia.com/cuda/wsl](https://developer.nvidia.com/cuda/wsl)

Minimum requirement:

* NVIDIA driver ≥ **535.x**

Verify on Windows:

```powershell
nvidia-smi
```

---

## 1. Base system setup (inside WSL)

Enter WSL:

```powershell
wsl -d Ubuntu-22.04
```

Update system:

```bash
sudo apt update && sudo apt upgrade -y
```

Install core build tools:

```bash
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    ca-certificates \
    software-properties-common \
    python3 \
    python3-pip \
    python3-venv
```

---

## 2. Install OpenMPI (system MPI)

For pedagogical MPI work, **Ubuntu’s OpenMPI is correct and stable**.

```bash
sudo apt install -y \
    openmpi-bin \
    libopenmpi-dev
```

Verify:

```bash
mpirun --version
```

Should see Open MPI 4.x.

---

## (Optional) 3. Install CUDA toolkit

This is not needed if we only want to use CUDA through Julia + CUDA.jl. Because CUDA.jl auto downloads a version-matched CUDA runtime. And if install system CUDA toolkit, some auto generated conf files might cause warnings while using CUDA.jl.

Check the conf files:

```bash
ls /etc/ld.so.conf.d | grep cuda
```

, we might want to rename them to disable them if said warnings appear.

### 3.1 DO NOT install Windows CUDA toolkit

Only the **driver** is needed on Windows. The **toolkit** lives inside WSL.

---

### 3.2 Add NVIDIA CUDA repository (inside WSL)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
```

```bash
sudo apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
```

```bash
sudo add-apt-repository \
    "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
```

```bash
sudo apt update
```

---

### 3.3 Install CUDA toolkit

There is a known and well-understood issue on Ubuntu 22.04 (jammy) when installing the CUDA toolkit from NVIDIA’s WSL repository. The short version is:

nsight-systems depends on libtinfo5, which does not exist in Ubuntu 22.04, and NVIDIA’s meta-package incorrectly pulls it in.

So do NOT run this:

```bash
sudo apt install -y cuda-toolkit-12-3
```

. Instead we install CUDA toolkit without Nsight.

First remove any partial CUDA state.

```bash
sudo apt purge -y cuda* nsight*
sudo apt autoremove -y
```

Then install CUDA toolkit components explicitly.

```bash
sudo apt install -y \
    cuda-compiler-12-3 \
    cuda-cudart-12-3 \
    cuda-cudart-dev-12-3 \
    cuda-libraries-12-3 \
    cuda-libraries-dev-12-3
```

This avoids Nsight and gives us:

- nvcc
- CUDA runtime
- CUDA math libraries

Everything CUDA.jl needs.

After that we can check with these 2 commands.

```bash
nvcc --version
nvidia-smi
```

If the first one gives a not-found like

```bash
Command 'nvcc' not found, but can be installed with: sudo apt install nvidia-cuda-toolkit
```

, it does not necessarily mean nvcc is not installed. It usually is because on WSL + NVIDIA repo, **`nvcc` is not placed on `/usr/bin`**, and lives here instead:

```bash
/usr/local/cuda-12.3/bin/nvcc
```

The shell simply does not have that directory on `PATH`.

So first we confirm `nvcc` exists

```bash
ls /usr/local/cuda-12.3/bin/nvcc
```

. If this directory exists and looks good, we can add CUDA to PATH ourselves.

```bash
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Reload.

```bash
source ~/.bashrc
```

Then verify again.

```bash
which nvcc
nvcc --version
```

Should see smth like this.

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

And if we want, can also verify this.

```bash
nvidia-smi
```
---

## 4. Install Julia

Assuming we are in non-root user home directory `~`. Take Julia 1.12.2 as example.

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-1.12.2-linux-x86_64.tar.gz
tar -xvf julia-1.12.2-linux-x86_64.tar.gz
```

We will get it at ~/julia-1.12.2/.

(Optional) Create symlink so we don't have to change the PATH when Julia version changes.

```bash
ln -s julia-1.12.2 julia
```

Then add to PATH.

```bash
export PATH="$HOME/julia/bin:$PATH"
source ~/.bashrc
```
---

## 5. Julia packages: MPI, CUDA, IJulia

Launch Julia:

```bash
julia
```

Inside Julia REPL:

```julia
using Pkg

Pkg.add([
    "MPI",
    "CUDA",
    "IJulia",
    "KernelAbstractions",
    "Distributed"
])
```

---

### 5.1 Configure MPI.jl to use system OpenMPI

Still in Julia:

```julia
using MPI
MPI.install_mpi_binary(; force=false)
```

Then explicitly select system MPI:

```bash
export JULIA_MPI_BINARY=system
```

Add this to `~/.bashrc`.

Verify:

```bash
julia -e 'using MPI; MPI.Init(); println(MPI.Comm_size(MPI.COMM_WORLD)); MPI.Finalize()'
```

---

### 5.2 Verify CUDA.jl

Inside Julia:

```julia
using CUDA
CUDA.versioninfo()
```

We should see:

* CUDA toolkit version
* Detected NVIDIA GPU
* `functional = true`

If not, stop here, do not proceed until this works.

---

## 6. Install JupyterLab

`pipx` is explicitly designed for **user-facing Python applications** like JupyterLab.

### 6.1. Install pipx (system-managed, safe)

```bash
sudo apt install -y pipx
```

Ensure it is on PATH:

```bash
pipx ensurepath
```

Restart the shell (or `source ~/.bashrc`).

---

### 6.2. Install JupyterLab via pipx

```bash
pipx install jupyterlab
```

This:

* Creates an isolated virtual environment
* Installs JupyterLab there
* Exposes the `jupyter-lab` command on the PATH
* Avoids breaking system Python

Add permanant PATH if we want.

```bash
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

Verify:

```bash
jupyter-lab --version
```

---

### 6.3. Register Julia with this Jupyter installation

Inside Julia:

```julia
using IJulia
IJulia.installkernel("Julia")
```

IJulia automatically detects `jupyter` from PATH (pipx makes this work).

---

### 6.4. Launch JupyterLab

```bash
jupyter-lab --no-browser
```

Open the printed URL in our **Windows browser**.