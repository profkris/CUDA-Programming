# CUDA Programming: Beginner to Advanced

A comprehensive collection of CUDA C/C++ programming examples covering GPU computing, parallel programming, GPU architecture, memory management, performance optimization, scientific computing and High Performance Computing (HPC).

## About this repository

Follow the numbered examples from basic kernels to GPU benchmarking. Run them on Google Colab, Kaggle, a local NVIDIA GPU system or a Slurm-managed HPC cluster. The original examples have been imported byte-for-byte; compilation, runtime testing and modernization are subsequent work.

## Repository structure

| Chapter | Topic | Files |
|---|---|---|
| 01 | [CUDA Fundamentals](01_CUDA_Fundamentals/) | 01–04 |
| 02 | [Threads, Blocks and Grids](02_Threads_Blocks_Grids/) | 05–12 |
| 03 | [Vector Operations](03_Vector_Operations/) | 13–24 |
| 04 | [GPU Memory Management](04_GPU_Memory_Management/) | 25–30 |
| 05 | [Matrix Multiplication](05_Matrix_Multiplication/) | 31–38 |
| 06 | [GPU Architecture](06_GPU_Architecture/) | 39–44 |
| 07 | [Occupancy and Throughput](07_Occupancy_and_Throughput/) | 45–47 |
| 08 | [GPU Performance Benchmarking](08_GPU_Benchmarking/) | 48–52 |
| 09 | [Scientific Computing](09_Scientific_Computing/) | 53–55 |
| 10 | [Practice Programs](10_Practice_Programs/) | 56–63 |
| 11 | [Assignments](11_Assignments/) | 64–70 |
| 12 | [Slurm and HPC](12_Slurm_HPC/) | 71–74 |

See [MIGRATION.md](MIGRATION.md) for old-to-new filenames and [the original README](docs/ORIGINAL_README.md) for the original instructions.

## 1. Prerequisites

Basic C/C++ and Linux command-line knowledge, a CUDA-compatible NVIDIA GPU, a compatible driver and the CUDA Toolkit (including `nvcc`). Colab and Kaggle provide remote GPU environments; a local GPU is not required. Accelerator availability and installed software vary.

## 2. Running CUDA programs on Google Colab

### Step 1: Open a notebook

Visit [Google Colab](https://colab.research.google.com/) and create a notebook.

### Step 2: Enable GPU runtime

Choose Runtime → Change runtime type → Hardware accelerator → GPU, then save.

### Step 3: Check GPU availability

Run in a notebook cell:

```bash
!nvidia-smi
```

For example, a Tesla T4 may report 15360 MiB memory, driver 550.90.07 and CUDA 12.4. The actual output depends on the allocated environment. If `nvidia-smi` is not found, check the accelerator setting and environment. The CUDA version in `nvidia-smi` indicates driver support, not necessarily the installed toolkit version.

### Step 4: Verify the compiler

```bash
!nvcc --version
```

### Step 5: Create a CUDA source file

Use `%%writefile` as the first line of a new cell:

```cpp
%%writefile test.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPU() {
    printf("Hello from GPU!\n");
}

int main() {
    printf("Hello from CPU!\n");
    helloGPU<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
```

### Step 6: Compile

```bash
!nvcc test.cu -o test
```

### Step 7: Execute

```bash
!./test
```

Expected output:

```text
Hello from CPU!
Hello from GPU!
```

### Step 8: Run repository examples

```bash
!git clone https://github.com/Sanket5153/CUDA-Programming.git
```

```python
%cd CUDA-Programming/01_CUDA_Fundamentals
```

```bash
!nvcc 01_hello_cpu_gpu.cu -o hello
!./hello
```


## 3. Running CUDA programs on Kaggle

1. Visit [Kaggle Notebooks](https://www.kaggle.com/code) and create a notebook.
2. Enable an available GPU accelerator in notebook settings.
3. Check the GPU and compiler:

```bash
!nvidia-smi
!nvcc --version
```

4. Enable internet access in notebook settings if permitted, then clone:

```bash
!git clone https://github.com/Sanket5153/CUDA-Programming.git
```

```python
%cd CUDA-Programming/01_CUDA_Fundamentals
```

5. Compile and execute:

```bash
!nvcc 01_hello_cpu_gpu.cu -o hello
!./hello
```

Alternatively, use the `%%writefile` method from the Colab section. If cloning is unavailable, upload the source manually.

## 4. Running CUDA programs on a local Linux system

Verify GPU and toolkit:

```bash
nvidia-smi
nvcc --version
```

If CUDA is installed under `/usr/local/cuda` but not on your PATH:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Clone, compile and run:

```bash
git clone https://github.com/Sanket5153/CUDA-Programming.git
cd CUDA-Programming/01_CUDA_Fundamentals
nvcc 01_hello_cpu_gpu.cu -o hello
./hello
```

The imported first example launches ten GPU threads; expect one CPU greeting and ten GPU greetings, not the single GPU greeting from the standalone notebook example.

## 5. Running CUDA programs on an HPC cluster

Login and discover CUDA modules:

```bash
ssh username@cluster-login-node
module avail cuda
module load cuda
nvcc --version
```

Inspect GPU partitions:

```bash
sinfo
sinfo -o "%P %D %N %G %t"
```

Clone the repository (or transfer it if external network access is restricted):

```bash
git clone https://github.com/Sanket5153/CUDA-Programming.git
cd CUDA-Programming/01_CUDA_Fundamentals
```

Request an interactive GPU allocation:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=4G --time=00:30:00 --pty bash
```

In the allocated shell:

```bash
module load cuda
nvidia-smi
nvcc 01_hello_cpu_gpu.cu -o hello
./hello
exit
```

Use your cluster's partition, account, QoS, GPU resource and module names. Avoid running GPU workloads on login nodes.

## 6. Slurm batch execution

Create `cuda_job.sh` in `01_CUDA_Fundamentals`:

```bash
#!/bin/bash
#SBATCH --job-name=CUDA-Test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=cuda_%j.out
#SBATCH --error=cuda_%j.err

set -e
module load cuda
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi
nvcc --version
nvcc 01_hello_cpu_gpu.cu -o hello
srun ./hello
```

Submit and monitor (replace `JOB_ID` with the actual job number):

```bash
sbatch cuda_job.sh
squeue -u "$USER"
scontrol show job JOB_ID
sacct -j JOB_ID
cat cuda_JOB_ID.out
scancel JOB_ID
```

Cancel only when needed. The `gpu` partition and `cuda` module are examples. See [Slurm and HPC](12_Slurm_HPC/) for original support scripts; review site-specific settings and old filenames before use.

## 7. GPU architecture-specific compilation

| GPU | Architecture | Compute capability |
|---|---|---|
| Tesla V100 | Volta | 7.0 |
| Tesla T4 | Turing | 7.5 |
| A100 / A30 | Ampere | 8.0 |
| RTX 3090 | Ampere | 8.6 |
| L4 / RTX 4090 | Ada Lovelace | 8.9 |
| H100 / H200 | Hopper | 9.0 |
| B200 | Blackwell | 10.0 |
| RTX PRO 6000 Blackwell | Blackwell | 12.0 |

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc -arch=sm_80 test.cu -o test
nvcc -arch=sm_90 test.cu -o test
nvcc -arch=sm_120 test.cu -o test
nvcc -arch=native test.cu -o test
```

Select one architecture appropriate to your GPU. The toolkit must support the target; `-arch=native` requires a compatible toolkit and visible GPU. On HPC systems, query the allocated compute node.

## 8. GPU monitoring

```bash
nvidia-smi
watch -n 2 nvidia-smi
nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 2
nvidia-smi dmon
```

Some metrics are unavailable on certain GPUs or restricted environments.

## 9. CUDA debugging and profiling

```bash
nvcc -g -G test.cu -o test
compute-sanitizer --tool memcheck ./test
compute-sanitizer --tool racecheck ./test
nsys profile -o cuda_report ./test
ncu ./test
```

Profiling tools must be installed, with appropriate permissions. Use optimized builds for performance comparisons; `-G` is for debugging.

## 10. Learning roadmap

Work through fundamentals, execution model, vectors, memory, matrices, architecture, occupancy, benchmarks, scientific computing and HPC. Planned additions: pinned/unified memory, coalescing, streams/events, atomics, reductions, prefix sums, multi-GPU programming, CUDA-aware MPI, cuBLAS/cuFFT/cuSOLVER and Tensor Cores.

## 11. Build and repository status

Build a selected standalone example from the repository root:

```bash
make SRC=01_CUDA_Fundamentals/01_hello_cpu_gpu.cu
./build/01_hello_cpu_gpu
make clean
```

The Makefile does not build every example automatically. Original exercises may be incomplete; some examples have legacy assumptions or require extra libraries. GPU compilation, expected numerical results and performance claims have not yet been validated. No new blanket license is assigned pending source ownership review.

## 12. Official references

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
- [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [Slurm Workload Manager](https://slurm.schedmd.com/)
- [Google Colab](https://colab.research.google.com/)
- [Kaggle Notebooks](https://www.kaggle.com/code)

## Author

**Dr Sabbi Vamshi Krishna**

Project Manager — HPC Technologies Group

Centre for Development of Advanced Computing (C-DAC), Pune, India

GitHub: https://github.com/profkris

---

Happy CUDA Programming!
