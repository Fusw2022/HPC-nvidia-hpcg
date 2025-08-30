# HPCG 基准测试优化
这是基于集群 V100 队列的 HPCG 基准测试优化，选择的实现是nvidia-hpcg，目前白天最高 270.816 GFLOP/s，夜晚（0点后）最高 268.86 GFLOP/s。按照 Note 所列建议进行了如下实验：
- 调整 HPCG 的输入配置文件：采用 256x256x256
- 尝试不同的编译器实现与版本：CXX = nvcc -ccbin icpx(qopenmp)
- 尝试不同的编译选项：-march=cascadelake，其余的使用原生（-Ofast；-funroll-loops等均启用）
- 尝试不同的数学库：使用原厂 nvhpc@25.1（cuda@12.8.0与此版本的HPCG并不兼容）
- 尝试不同的 MPI 实现：使用Intel MPI（要求编译器后端使用icpx）
- 尝试不同的 MPI 运行参数（例如进程数和核心数）：--cpus-per-task=4；启用bind-to core

# HPCG 使用方法
目前代码存放在 https://github.com/Fusw2022/HPC-nvidia-hpcg 下，构建脚本为build_sample.sh，运行脚本为run.sh。
1. 【下载】git clone https://github.com/Fusw2022/HPC-nvidia-hpcg
2. 【进入】下载到集群后，进入 HPC-nvidia-hpcg 文件夹
3. 【编译】./build_sample.sh "githash" 3 1 1 1 0 1 0
4. 【运行】sbatch run.sh
若项目已经编译成功，可能需要首先 rm -rf build。

# HPCG 实践过程

## 移植到集群
- 下载 nvidia-hpcg 代码
- 调整 build_sample.sh 路径设置，按照 run_sample.sh 编写 run.sh 脚本并相应调整路径设置
- 设置 makefile 中 arch = CUDA_X86
- 调整 Make.CUDA_X86 中 CUDA_ARCH 为 CUDA_ARCH = -gencode=arch=compute_70,code=sm_70 以适配 V100 架构

## 数据测试
- 调整 HPCG 的输入配置文件：更改 run.sh 中 nx, ny, nz
- 尝试不同的编译器/mpi实现：修改 Make.CUDA_X86 中的 CXX 与 CXXFLAGS，并在必要时修改 build_sample.sh 和 run.sh 中的路径及相应命令
- 尝试不同的编译选项：修改 Make.CUDA_X86 中的 CPU_ARCH 与 CXXFLAGS
- 尝试不同的 MPI 运行参数：修改 run.sh 中 sbatch 部分


<!-- 下面是 nvidia-hpcg 自带 README 中文版-->



# NVIDIA 高性能共轭梯度基准测试（HPCG）

NVIDIA HPCG 基于 [HPCG](https://github.com/hpcg-benchmark/hpcg) 基准测试开发，并针对 NVIDIA 加速的高性能计算系统进行了性能优化。

NVIDIA 的 HPCG 基准测试对高性能共轭梯度（HPCG）基准测试进行了加速。HPCG 是一个软件包，它使用双精度（64 位）浮点值执行固定数量的多重网格预处理（使用对称高斯 - 赛德尔平滑器）共轭梯度（PCG）迭代。

## 主要特点
* NVIDIA HPCG 基准测试利用 NVIDIA 高性能数学库：[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) 和 [NVPL Sparse](https://docs.nvidia.com/nvpl/_static/sparse/index.html)，在 NVIDIA GPU 和 Grace CPU 上实现稀疏矩阵 - 向量乘法（SpMV）和稀疏矩阵三角求解器（SpSV）的最高可能性能。

* NVIDIA HPCG 基准测试支持高度可配置的命令行参数，用于决定：
    * GPU 和 Grace CPU 的问题规模
    * 3D 秩网格形状
    * 执行模式：仅 CPU、仅 GPU 和异构模式
    * 点对点通信：MPI_Host（Send/Recv）、MPI_Host_Alltoallv、MPI_CUDA_Aware、MPI_CUDA_Aware_Alltoallv 和 NCCL
    * NUMA 相关配置

    详细说明参见 `bin/RUNNING-x86` 和 `bin/RUNNING-aarch64`。
* NVIDIA HPCG 基准测试支持的稀疏存储格式是标准的 [切片 ELLPACK 格式（SELL）](https://docs.nvidia.com/cuda/cusparse/#sliced-ellpack-sell)。

## 支持的平台
* NVIDIA HPCG 基准测试支持在具有 NVIDIA Ampere GPU 架构（sm80）和 NVIDIA Hopper GPU 架构（sm90）的 x86 和 NVIDIA Grace CPU 系统上进行仅 GPU 执行，支持在 NVIDIA Grace CPU 上进行仅 CPU 执行，以及在 NVIDIA Grace Hopper 超级芯片上进行 GPU - Grace 异构执行。
* NVIDIA HPCG 仅支持 Linux 操作系统。

## 先决条件
* Git
* MPI，OpenMPI 4.1+ 和 MPICH 4.0+
* CUDA 工具包 12.3+，用于 NVIDIA GPU 执行
* cuSPARSE 12.3+，用于 NVIDIA GPU 执行
* cuBLAS 12.2+，用于 NVIDIA GPU 执行
* GCC 13.0+，用于 NVIDIA Grace CPU 执行
* NVPL 24.03+，用于 NVIDIA Grace CPU 执行
* NCCL 2.19+，可选，用于进程间通信

## 编译和构建
### 克隆仓库
SSH 方式
```
git clone ssh://github.com/NVIDIA/nvidia-hpcg
```
HTTPS 方式
```
git clone https://github.com/NVIDIA/nvidia-hpcg
```
GitHub 命令行方式
```
gh repo clone NVIDIA/nvidia-hpcg
```

### 编译 NVIDIA HPCG 基准测试
`build_sample.sh` 脚本可用于编译和构建 NVIDIA HPCG 基准测试。在运行 `make` 命令之前，必须将 MPI、CUDA 工具包、CUDA 数学库、NCCL 和 NVPL Sparse 的路径导出到 `MPI_PATH`、`CUDA_PATH`、`MATHLIBS_PATH`、`NCCL_PATH` 和 `NVPL_SPARSE_PATH` 中。
可使用以下选项决定目标平台：
* `USE_CUDA`，设为 1 时为 NVIDIA GPU 构建，否则设为 0。
* `USE_GRACE`，设为 1 时为 NVIDIA Grace CPU 构建，否则设为 0。设为 0 时，代码为 x86 平台构建。
* `USE_NCCL`，设为 1 时为 NCCL 构建，否则设为 0。

`USE_CUDA` 和 `USE_GRACE` 选项用于创建支持以下三种执行模式之一的二进制文件：
* 对于仅 GPU 模式，将 `USE_CUDA` 设为 1。当 `USE_GRACE=1` 时，为 `aarch64` 构建；当 `USE_GRACE=0` 时，为 `x86` 构建。
* 对于仅 Grace 模式，将 `USE_CUDA` 设为 0 且 `USE_GRACE` 设为 1。
* 对于 GPU - Grace 模式，将 `USE_CUDA` 设为 1 且 `USE_GRACE` 设为 1。

`build_sample.sh` 脚本使用 `setup/MAKE.CUDA_AARCH64` 和 `setup/MAKE.CUDA_X86` 来组成 `make` 命令的包含和链接行。这两个脚本定义了源代码中使用的编译时选项。这些选项在两个 `setup/MAKE.CUDA_*` 脚本文件中有说明。构建脚本创建 `build` 目录，并将 NVIDIA HPCG 二进制文件存储在 `build/bin` 和 `bin` 目录中（二进制文件从 `build/bin` 复制到 `bin`）。构建脚本可以创建以下二进制文件之一：
* 当 `USE_CUDA=1` 时，生成 xhpcg。
* 当 `USE_CUDA=0` 且 `USE_GRACE=1` 时，生成 xhpcg - cpu。


## 运行 NVIDIA HPCG 基准测试
NVIDIA HPCG 基准测试使用与标准 HPCG 基准测试相同的输入格式，用户也可以借助选项传递基准测试参数。请参见 HPCG 基准测试以了解 HPCG 软件概念和最佳实践的入门知识。`bin` 目录包含运行 NVIDIA HPCG 基准测试的脚本以及说明和示例。文件 `bin/RUNNING-x86` 和 `bin/RUNNING-aa64` 分别详细解释了如何在 `x86` 和 `aarch64` 平台上运行 NVIDIA HPCG 基准测试。`run_sample.sh` 脚本提供了四个在 `x86` 和 Grace Hopper x4 平台上运行的示例。

### 深入了解异构（GPU - GRACE）执行模式
NVIDIA HPCG 基准测试可以在包含 GPU 和 Grace CPU 的异构系统（如 GRACE HOPPER）上高效运行。该方法包括为每个 GPU 分配一个 MPI 秩，并为 Grace CPU 分配一个或多个 MPI 秩。由于 GPU 的性能明显快于 Grace CPU，策略是为 GPU 分配比 Grace CPU 更大的本地问题规模。这确保了在像 `MPI_Allreduce` 这样的 MPI 阻塞通信步骤中，GPU 的执行不会被 Grace CPU 较慢的执行中断。

在 NVIDIA HPCG 基准测试中，GPU 和 Grace 的本地问题被配置为仅在一个维度上不同，而其他维度保持相同。这种设计能够在 GPU 和 Grace 秩之间相同的维度上进行适当的光晕交换操作。下图展示了这种设计的一个示例。GPU 和 Grace 秩具有相同的 x 和 y 维度，光晕交换就在这里进行。z 维度不同，这使得可以为 GPU 和 Grace 秩分配不同的本地问题。NVIDIA HPCG 基准测试可以灵活选择秩的 3D 形状、选择不同的维度以及配置 GPU 和 Grace 秩的大小。更多细节请参考 `bin/RUNNING-aarch64`。

<img src="images/hpcg-gpu-grace-example.png" alt="drawing" width="150"/>

### 结果解释
默认情况下，NVIDIA HPCG 基准测试将日志输出到标准输出（`stdout`）。要打印到日志文件中，请将旋钮 `--of` 设置为 1。
即使运行是有效的，在性能日志中也有一些性能标志需要注意（行号是针对输出到日志文件的情况）：
* 在迭代摘要部分（第 68 行），每组优化的 CG 迭代次数（第 72 行）应尽可能接近参考值（即 50 次迭代）。用户可以尝试不同的参数，如本地问题规模和 3D 网格形状，以实现较少的迭代次数。
* 在 GFLOP/s 摘要（第 100 行）中，第 107 行的“包含收敛和优化阶段开销的总计”值应尽可能接近“原始总计”。否则，要确保优化的 CG 迭代次数、设置时间（第 20 行）和优化时间（第 82 行）与总执行时间相比是合理的。这在多节点扩展时很重要。
* 在多节点平台上扩展时，大部分 DDOT 计算时间是 `MPI_Allreduce` 的时间。较高的 `MPI_Allreduce` 时间表明由于本地问题规模小或配置或平台存在问题而导致扩展瓶颈。

## 支持
如有问题或要提供反馈，请联系 [HPCBenchmarks@nvidia.com](mailto:HPCBenchmarks@nvidia.com)

## 许可证
许可证文件可在 [LICENSE](LICENSE) 文件中找到。