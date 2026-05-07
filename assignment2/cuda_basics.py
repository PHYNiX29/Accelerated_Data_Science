"""
Assignment 2 (UCS547) - CUDA C++ Programming Basics
Designed for Google Colab with NVIDIA GPU runtime.
"""

import os
import subprocess
import time
import numpy as np

# =============================================================================
# Q1. Identify !, %, and %% used in cells in Google Colab
# =============================================================================

print("=" * 70)
print("Q1. Special Prefixes in Google Colab Cells")
print("=" * 70)

q1_explanation = """
Google Colab supports special prefixes that let you run non-Python commands
directly inside notebook cells:

1. ! (Exclamation Mark) - Shell Command Prefix
   - Used at the beginning of a line to run a single shell (bash) command.
   - Each '!' command runs in a temporary subshell; changes like 'cd' do
     not persist across lines.
   - Examples:
       !ls -la
       !pip install numpy
       !nvcc --version
       !nvidia-smi

2. % (Single Percent) - Line Magic Command
   - Applies a built-in IPython 'magic' command to a single line.
   - These are special commands provided by IPython/Jupyter for common tasks.
   - Examples:
       %cd /content          -> Change directory (persistent)
       %timeit x = sum(range(1000))  -> Time a single statement
       %pwd                  -> Print working directory
       %env MY_VAR=hello     -> Set an environment variable
       %matplotlib inline    -> Enable inline plotting

3. %% (Double Percent) - Cell Magic Command
   - Applies a magic command to the entire cell (not just one line).
   - Must appear as the very first line of the cell.
   - Examples:
       %%time               -> Time the execution of the whole cell
       %%writefile test.cu  -> Write entire cell contents to a file
       %%bash               -> Run the entire cell as a bash script
       %%html               -> Render the cell content as HTML
       %%capture             -> Capture and suppress cell output

Key Differences:
   - '!'  runs a shell command in a subprocess (temporary).
   - '%'  runs a single-line IPython magic (persistent session state).
   - '%%' runs a cell-level IPython magic (applies to entire cell body).
"""
print(q1_explanation)


# =============================================================================
# Q2. Identify key nvidia-smi commands with multiple options
# =============================================================================

print("=" * 70)
print("Q2. Key nvidia-smi Commands and Options")
print("=" * 70)

q2_explanation = """
nvidia-smi (NVIDIA System Management Interface) is a command-line tool for
monitoring and managing NVIDIA GPU devices.

Common Commands and Options:
------------------------------------------------------------
1. nvidia-smi
   - Default command; shows a summary table of all GPUs including:
     GPU name, temperature, fan speed, power usage, memory usage,
     GPU utilization, and running processes.

2. nvidia-smi -L
   - Lists all available GPUs with their UUIDs.
   - Example output: GPU 0: Tesla T4 (UUID: GPU-xxxxx...)

3. nvidia-smi -q
   - Displays detailed (query) information about all GPUs.
   - Includes clocks, memory, ECC errors, power, temperature, etc.

4. nvidia-smi -q -d MEMORY
   - Queries only the MEMORY section of detailed GPU info.
   - Other sections: UTILIZATION, TEMPERATURE, POWER, CLOCK, ECC, etc.

5. nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu --format=csv
   - Custom query with CSV-formatted output.
   - Useful for scripting and logging GPU metrics.

6. nvidia-smi dmon -s u -d 1
   - Device monitoring mode: streams utilization stats every 1 second.
   - The -s flag selects metrics (u=utilization, p=power, m=memory, etc.).

7. nvidia-smi pmon -i 0 -s u -d 1
   - Process monitoring mode: tracks per-process GPU utilization.

8. nvidia-smi -pm 1
   - Enables persistence mode (keeps driver loaded for faster responses).

9. nvidia-smi -r -i 0
   - Resets GPU 0 (useful for recovering from errors).

10. nvidia-smi topo -m
    - Shows the GPU topology matrix (NVLink/PCIe interconnects).
"""
print(q2_explanation)

print("\n--- Running 'nvidia-smi' on this system ---\n")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    print(result.stdout if result.stdout else result.stderr)
except FileNotFoundError:
    print("[INFO] nvidia-smi not found. This command requires an NVIDIA GPU and drivers.")
except subprocess.TimeoutExpired:
    print("[INFO] nvidia-smi timed out.")

print("\n--- Running 'nvidia-smi -L' (List GPUs) ---\n")
try:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    print(result.stdout if result.stdout else result.stderr)
except FileNotFoundError:
    print("[INFO] nvidia-smi not found.")
except subprocess.TimeoutExpired:
    print("[INFO] nvidia-smi timed out.")

print("\n--- Running 'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv' ---\n")
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
        capture_output=True, text=True, timeout=10
    )
    print(result.stdout if result.stdout else result.stderr)
except FileNotFoundError:
    print("[INFO] nvidia-smi not found.")
except subprocess.TimeoutExpired:
    print("[INFO] nvidia-smi timed out.")


# =============================================================================
# Q3. Debug common CUDA errors
# =============================================================================

print("\n" + "=" * 70)
print("Q3. Debugging Common CUDA Errors")
print("=" * 70)

q3_explanation = """
CUDA programs can fail silently or produce confusing errors. Below are the
three most common categories of bugs and how to debug them.

---------------------------------------------------------------------------
Error 1: Zero Output (Kernel Produces No Output)
---------------------------------------------------------------------------
Symptom:
    Program compiles and runs without errors, but prints nothing or all zeros.

Common Causes:
    a) Missing cudaDeviceSynchronize() after kernel launch.
       - GPU kernels are asynchronous. Without synchronization, the host may
         exit or read results before the kernel finishes.
       - Fix: Add cudaDeviceSynchronize() after every kernel launch.

    b) Forgetting cudaMemcpy() to copy results back from device to host.
       - The kernel writes to device memory; you must copy it back.
       - Fix: cudaMemcpy(host_ptr, dev_ptr, size, cudaMemcpyDeviceToHost);

    c) Launching kernel with 0 blocks or 0 threads.
       - kernel<<<0, 256>>>() launches nothing.
       - Fix: Verify grid and block dimensions are > 0.

    Example (buggy):
        myKernel<<<1, 256>>>(d_data);
        // Missing: cudaDeviceSynchronize();
        // Missing: cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
        // Result: h_data is all zeros.

---------------------------------------------------------------------------
Error 2: Incorrect Indexing (Out-of-Bounds Access)
---------------------------------------------------------------------------
Symptom:
    Garbled output, partial results, or program crash (illegal memory access).

Common Causes:
    a) Wrong global index formula.
       - Correct 1D: idx = blockIdx.x * blockDim.x + threadIdx.x
       - Common mistake: using only threadIdx.x (ignores block offset).

    b) No bounds checking.
       - If array has N elements but total threads > N, extra threads access
         invalid memory.
       - Fix: if (idx < N) { /* safe work */ }

    c) Wrong grid/block dimensions for multi-dimensional data.
       - 2D indexing: row = blockIdx.y * blockDim.y + threadIdx.y
                      col = blockIdx.x * blockDim.x + threadIdx.x

    Example (buggy):
        __global__ void add(int *a, int *b, int *c) {
            int i = threadIdx.x;  // BUG: ignores blockIdx.x
            c[i] = a[i] + b[i];
        }
        // With 2 blocks of 256 threads, block 1 overwrites block 0's work.

---------------------------------------------------------------------------
Error 3: PTX / Compilation Errors
---------------------------------------------------------------------------
Symptom:
    nvcc reports errors like "ptxas error", "invalid device function", or
    the program crashes at kernel launch.

Common Causes:
    a) Architecture mismatch.
       - Compiling for sm_50 but running on sm_75 hardware (or vice versa).
       - Fix: nvcc -arch=sm_75 program.cu (match your GPU).
       - On Colab (Tesla T4): use -arch=sm_75.

    b) Using unsupported features for the target architecture.
       - Example: __half (FP16) operations need sm_53+.
       - Fix: Compile with appropriate -arch flag.

    c) Syntax errors in device code.
       - Missing __global__ or __device__ qualifiers.
       - Using host-only functions (like printf in very old archs) in kernels.
       - Fix: Use correct qualifiers and check CUDA version compatibility.

    d) Kernel launch failure (silent).
       - Always check: cudaError_t err = cudaGetLastError();
         if (err != cudaSuccess) printf("Error: %s\\n", cudaGetErrorString(err));

Debugging Tools:
    - cuda-memcheck / compute-sanitizer: Detects memory errors at runtime.
    - nvcc -lineinfo: Adds source line info for better error messages.
    - CUDA_LAUNCH_BLOCKING=1: Forces synchronous kernel launches for debugging.
"""
print(q3_explanation)


# =============================================================================
# Q4. CUDA C/C++ Program - GPU Kernel Execution and Thread Indexing
# =============================================================================

print("\n" + "=" * 70)
print("Q4. CUDA Kernel Execution and Thread Indexing")
print("=" * 70)
print("Writing, compiling, and running a CUDA program...")
print("  - 1 block, 8 threads")
print("  - Each thread prints: Hello from GPU thread <global_thread_id>")
print("  - global_thread_id = blockIdx.x * blockDim.x + threadIdx.x\n")

q4_cuda_code = r"""
#include <stdio.h>
#include <cuda_runtime.h>

// ---- Device Code ----
__global__ void helloKernel() {
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d\n", global_thread_id);
}

// ---- Host Code ----
int main() {
    printf("Launching kernel with 1 block and 8 threads...\n\n");

    // Launch kernel: 1 block, 8 threads per block
    helloKernel<<<1, 8>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\nKernel execution complete.\n");
    return 0;
}
"""

q4_cu_path = "/tmp/q4_thread_index.cu"
q4_bin_path = "/tmp/q4_thread_index"

try:
    with open(q4_cu_path, "w") as f:
        f.write(q4_cuda_code)
    print(f"[INFO] CUDA source written to {q4_cu_path}")

    compile_result = subprocess.run(
        ["nvcc", "-o", q4_bin_path, q4_cu_path],
        capture_output=True, text=True, timeout=60
    )
    if compile_result.returncode == 0:
        print("[INFO] Compilation successful.")
        run_result = subprocess.run(
            [q4_bin_path], capture_output=True, text=True, timeout=10
        )
        print("\n--- Program Output ---")
        print(run_result.stdout)
        if run_result.stderr:
            print(run_result.stderr)
    else:
        print("[ERROR] Compilation failed:")
        print(compile_result.stderr)
except FileNotFoundError:
    print("[INFO] nvcc not found. This requires CUDA toolkit (available on Colab GPU runtime).")
    print("\nExpected output when run on Colab:")
    print("  Launching kernel with 1 block and 8 threads...")
    for i in range(8):
        print(f"  Hello from GPU thread {i}")
    print("  Kernel execution complete.")
except subprocess.TimeoutExpired:
    print("[INFO] Compilation or execution timed out.")


# =============================================================================
# Q5. CUDA Program - Host/Device Memory Separation
# =============================================================================

print("\n" + "=" * 70)
print("Q5. Host/Device Memory Separation")
print("=" * 70)
print("Writing, compiling, and running a CUDA memory management program...")
print("  - Create int array of size 5 on host")
print("  - cudaMalloc + cudaMemcpy host -> device")
print("  - Kernel prints values from device memory")
print("  - Copy back to host and print on CPU\n")

q5_cuda_code = r"""
#include <stdio.h>
#include <cuda_runtime.h>

#define N 5

// ---- Device Code ----
__global__ void printFromDevice(int *d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        printf("Device: d_arr[%d] = %d\n", idx, d_arr[idx]);
    }
}

// ---- Host Code ----
int main() {
    // Step 1: Create and initialize host array
    int h_arr[N] = {10, 20, 30, 40, 50};

    printf("Step 1: Host array initialized.\n");
    for (int i = 0; i < N; i++) {
        printf("  h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Step 2: Allocate device memory
    int *d_arr;
    cudaMalloc((void **)&d_arr, N * sizeof(int));
    printf("\nStep 2: Device memory allocated with cudaMalloc.\n");

    // Step 3: Copy data from host to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    printf("Step 3: Data copied from host to device (cudaMemcpyHostToDevice).\n");

    // Step 4: Launch kernel to print values from device memory
    printf("\nStep 4: Kernel printing values from device memory:\n");
    printFromDevice<<<1, N>>>(d_arr, N);
    cudaDeviceSynchronize();

    // Step 5: Copy data back from device to host
    int h_result[N] = {0};
    cudaMemcpy(h_result, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nStep 5: Data copied back from device to host (cudaMemcpyDeviceToHost).\n");

    // Step 6: Print results on CPU
    printf("\nStep 6: Verifying data on CPU after round-trip:\n");
    for (int i = 0; i < N; i++) {
        printf("  h_result[%d] = %d\n", i, h_result[i]);
    }

    // Cleanup
    cudaFree(d_arr);
    printf("\nDevice memory freed. Done.\n");

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
"""

q5_cu_path = "/tmp/q5_memory.cu"
q5_bin_path = "/tmp/q5_memory"

try:
    with open(q5_cu_path, "w") as f:
        f.write(q5_cuda_code)
    print(f"[INFO] CUDA source written to {q5_cu_path}")

    compile_result = subprocess.run(
        ["nvcc", "-o", q5_bin_path, q5_cu_path],
        capture_output=True, text=True, timeout=60
    )
    if compile_result.returncode == 0:
        print("[INFO] Compilation successful.")
        run_result = subprocess.run(
            [q5_bin_path], capture_output=True, text=True, timeout=10
        )
        print("\n--- Program Output ---")
        print(run_result.stdout)
        if run_result.stderr:
            print(run_result.stderr)
    else:
        print("[ERROR] Compilation failed:")
        print(compile_result.stderr)
except FileNotFoundError:
    print("[INFO] nvcc not found. This requires CUDA toolkit (available on Colab GPU runtime).")
    print("\nExpected output when run on Colab:")
    print("  Step 1: Host array initialized.")
    print("    h_arr[0] = 10, h_arr[1] = 20, h_arr[2] = 30, h_arr[3] = 40, h_arr[4] = 50")
    print("  Step 2: Device memory allocated with cudaMalloc.")
    print("  Step 3: Data copied from host to device.")
    print("  Step 4: Kernel printing values from device memory:")
    for i, v in enumerate([10, 20, 30, 40, 50]):
        print(f"    Device: d_arr[{i}] = {v}")
    print("  Step 5: Data copied back from device to host.")
    print("  Step 6: Verifying data on CPU after round-trip:")
    for i, v in enumerate([10, 20, 30, 40, 50]):
        print(f"    h_result[{i}] = {v}")
    print("  Device memory freed. Done.")
except subprocess.TimeoutExpired:
    print("[INFO] Compilation or execution timed out.")


# =============================================================================
# Q6. Compare CPU Times - List/Tuple vs NumPy Arrays
# =============================================================================

print("\n" + "=" * 70)
print("Q6. CPU Time Comparison: List/Tuple vs NumPy Arrays")
print("=" * 70)

SIZE = 1_000_000
ITERATIONS = 10

print(f"\nArray size: {SIZE:,} elements")
print(f"Iterations per test: {ITERATIONS}")
print(f"Operation: Element-wise addition of two arrays\n")

# --- List Addition ---
list_a = list(range(SIZE))
list_b = list(range(SIZE))

list_times = []
for _ in range(ITERATIONS):
    start = time.perf_counter()
    list_result = [a + b for a, b in zip(list_a, list_b)]
    end = time.perf_counter()
    list_times.append(end - start)

avg_list_time = sum(list_times) / len(list_times)
print(f"Python List addition   : {avg_list_time:.6f} seconds (avg of {ITERATIONS} runs)")

# --- Tuple Addition ---
tuple_a = tuple(range(SIZE))
tuple_b = tuple(range(SIZE))

tuple_times = []
for _ in range(ITERATIONS):
    start = time.perf_counter()
    tuple_result = tuple(a + b for a, b in zip(tuple_a, tuple_b))
    end = time.perf_counter()
    tuple_times.append(end - start)

avg_tuple_time = sum(tuple_times) / len(tuple_times)
print(f"Python Tuple addition  : {avg_tuple_time:.6f} seconds (avg of {ITERATIONS} runs)")

# --- NumPy Array Addition ---
np_a = np.arange(SIZE)
np_b = np.arange(SIZE)

numpy_times = []
for _ in range(ITERATIONS):
    start = time.perf_counter()
    np_result = np_a + np_b
    end = time.perf_counter()
    numpy_times.append(end - start)

avg_numpy_time = sum(numpy_times) / len(numpy_times)
print(f"NumPy Array addition   : {avg_numpy_time:.6f} seconds (avg of {ITERATIONS} runs)")

# --- Summary ---
print("\n" + "-" * 50)
print("Performance Summary:")
print("-" * 50)
print(f"  List   : {avg_list_time:.6f} s")
print(f"  Tuple  : {avg_tuple_time:.6f} s")
print(f"  NumPy  : {avg_numpy_time:.6f} s")

speedup_list = avg_list_time / avg_numpy_time if avg_numpy_time > 0 else float('inf')
speedup_tuple = avg_tuple_time / avg_numpy_time if avg_numpy_time > 0 else float('inf')

print(f"\n  NumPy is ~{speedup_list:.1f}x faster than Python Lists")
print(f"  NumPy is ~{speedup_tuple:.1f}x faster than Python Tuples")

print("""
Why NumPy is faster:
  - NumPy uses contiguous C arrays in memory (cache-friendly).
  - Operations are vectorized in compiled C/Fortran (no Python loop overhead).
  - Python lists/tuples store pointers to individual objects, causing memory
    fragmentation and per-element type checking overhead.
  - NumPy leverages SIMD instructions for bulk arithmetic operations.
""")

print("=" * 70)
print("Assignment 2 Complete.")
print("=" * 70)
