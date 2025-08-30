#! /usr/bin/env bash

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

source /pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/intel-oneapi-compilers-2025.0.4-l4ofpzkv4tomad2jkp4pe5vxuvmjnsfi/setvars.sh

# Path to your NVHPC installation
export NVHPC_BASE=/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/nvhpc-25.1-gfpvhsdurdxu5qqwgkxsn6m76eohxn25/Linux_x86_64/25.1

# Set paths to components within NVHPC
export CXX_PATH=${NVHPC_BASE}/compilers
export CUDA_PATH=${NVHPC_BASE}/cuda

# --- CORRECTED PATHS START HERE ---
# Select a specific MPI implementation from your NVHPC installation
# export MPI_PATH=${NVHPC_BASE}/comm_libs/12.6/openmpi4/openmpi-4.1.5
# export MPI_PATH=${NVHPC_BASE}/comm_libs/12.6/hpcx/hpcx-2.21/ompi
export MPI_PATH=/pxe/opt/spack/opt/spack/linux-debian12-haswell/gcc-12.2.0/intel-oneapi-mpi-2021.14.1-jdda552mqvxz4g6vuwkboc7biptbtgge/mpi/2021.14

# NCCL is also located in a version-specific directory
export NCCL_PATH=${NVHPC_BASE}/comm_libs/12.6/nccl
# --- CORRECTED PATHS END HERE ---

export MATHLIBS_PATH=${NVHPC_BASE}/math_libs
export NVPL_SPARSE_PATH=${NVHPC_BASE}/math_libs

# Update PATH and LD_LIBRARY_PATH
export PATH=${CXX_PATH}/bin:${CUDA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MPI_PATH}/lib:${NCCL_PATH}/lib:${MATHLIBS_PATH}/lib:${LD_LIBRARY_PATH}


#xhpcg binary will be located in build/bin
mkdir -p build
cd build

######## USE Nvidia GPU? ############
# 1:         Yes
# O:         No 
export USE_CUDA=1
if [[ $5 == "0" ]]; then
    export USE_CUDA=0
fi
################################################

######## USE Grace CPU? ############
# 1:         Yes
# O:         No 
export USE_GRACE=0
if [[ $6 == "1" ]]; then
    export USE_GRACE=1
fi
################################################

######## USE NCCL? ############
# 1:         Yes
# O:         No 
export USE_NCCL=1
if [[ $7 == "0" ]]; then
    export USE_NCCL=0
fi
################################################

if [[ $USE_GRACE == 1 ]]; then
    ../configure CUDA_AARCH64
else
    ../configure CUDA_X86
fi

export build_B100=0
if [[ $8 == "1" ]]; then
    export build_B100=1
fi

make -j 16 \
    USE_CUDA=${USE_CUDA} \
    USE_GRACE=${USE_GRACE} \
    USE_NCCL=${USE_NCCL} \
    MPdir=${MPI_PATH} \
    MPlib=${MPI_PATH}/lib \
    Mathdir=${MATHLIBS_PATH} \
    NCCLdir=${NCCL_PATH} \
    CUDA_HOME=${CUDA_PATH} \
    NVPL_PATH=${NVPL_SPARSE_PATH} \
    HPCG_ENG_VERSION=${is_ENG_VERSION} \
    HPCG_COMMIT_HASH=$2 \
    HPCG_VER_MAJOR=$3 \
    HPCG_VER_MINOR=$4 \
    BUILD_B100=${build_B100}

#Move build/bin/xhpcg to bin/xhpcg
make install