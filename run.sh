#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Path to NVHPC installation
export NVHPC_BASE=/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/nvhpc-25.1-gfpvhsdurdxu5qqwgkxsn6m76eohxn25/Linux_x86_64/25.1

# Set paths to components within NVHPC
export CXX_PATH=${NVHPC_BASE}/compilers
export CUDA_PATH=${NVHPC_BASE}/cuda
# export MPI_PATH=${NVHPC_BASE}/comm_libs/12.6/openmpi4/openmpi-4.1.5
# export MPI_PATH=${NVHPC_BASE}/comm_libs/12.6/hpcx/hpcx-2.21/ompi
export MPI_PATH=/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/intel-oneapi-mpi-2021.14.1-jdda552mqvxz4g6vuwkboc7biptbtgge/mpi/2021.14
export NCCL_PATH=${NVHPC_BASE}/comm_libs/12.6/nccl
export MATHLIBS_PATH=${NVHPC_BASE}/math_libs
export NVPL_SPARSE=${NVHPC_BASE}/math_libs # NVPL is in math_libs

# Set the correct CUDA version for the math library paths
export CUDA_BLAS_VERSION=12.6

# Update PATH and LD_LIBRARY_PATH for runtime
export PATH=${CUDA_PATH}/bin:${MPI_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MPI_PATH}/lib:${NCCL_PATH}/lib:${MATHLIBS_PATH}/${CUDA_BLAS_VERSION}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${NVPL_SPARSE}/${CUDA_BLAS_VERSION}/lib64:${LD_LIBRARY_PATH}

dir="bin/"

echo "--- Running on ${SLURM_NTASKS} GPU ---"
# Local problem size
nx=256
ny=256
nz=256
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# mpirun ${ext} -np $SLURM_NTASKS --bind-to core ${dir}/xhpcg --nx $nx --ny $ny --nz $nz --rt 30
mpirun -np $SLURM_NTASKS ${dir}/xhpcg --nx $nx --ny $ny --nz $nz --rt 1830