<!-- TOC -->

- [How do we usually use threadIdx()](#how-do-we-usually-use-threadidx)
    - [1. What `threadIdx().x` actually is](#1-what-threadidxx-actually-is)
    - [2. Why we use it](#2-why-we-use-it)
    - [3. Applying this to the Blelloch scan kernel](#3-applying-this-to-the-blelloch-scan-kernel)
    - [4. How CUDA executes this](#4-how-cuda-executes-this)
    - [5. Small example to illustrate](#5-small-example-to-illustrate)
    - [6. Summary](#6-summary)
- [The x y z of threadIdx()](#the-x-y-z-of-threadidx)
    - [1. CUDA threads are indexed in 3 dimensions](#1-cuda-threads-are-indexed-in-3-dimensions)
    - [2. Why `.x` exists even if we only use 1D threads](#2-why-x-exists-even-if-we-only-use-1d-threads)
    - [3. Analogy with arrays](#3-analogy-with-arrays)
    - [4. Why CUDA uses x/y/z instead of i/j/k](#4-why-cuda-uses-xyz-instead-of-ijk)
    - [5. Important Julia-specific detail (1-based indexing)](#5-important-julia-specific-detail-1-based-indexing)
    - [Summary](#summary)
- [What does syncthreads() do](#what-does-syncthreads-do)
    - [1. What `syncthreads()` guarantees](#1-what-syncthreads-guarantees)
    - [2. Scope: block-level only](#2-scope-block-level-only)
    - [3. Why it is mandatory in our Blelloch scan](#3-why-it-is-mandatory-in-our-blelloch-scan)
    - [4. What would break without it](#4-what-would-break-without-it)
    - [5. Correctness rule (very important)](#5-correctness-rule-very-important)
    - [6. Performance implications](#6-performance-implications)
    - [7. Summary](#7-summary)
- [GPU specific parallel pattern](#gpu-specific-parallel-pattern)
- [Why is GPU parallel so different than CPU](#why-is-gpu-parallel-so-different-than-cpu)
    - [1. CPU vs GPU threads - they are fundamentally different](#1-cpu-vs-gpu-threads---they-are-fundamentally-different)
        - [1.1 CPU threads](#11-cpu-threads)
        - [1.2 GPU “threads”](#12-gpu-threads)
    - [2. Why the GPU code explicitly has upsweep + downsweep](#2-why-the-gpu-code-explicitly-has-upsweep--downsweep)
        - [GPU-friendly algorithm: Blelloch scan](#gpu-friendly-algorithm-blelloch-scan)
    - [3. Why the CPU version does NOT need upsweep/downsweep](#3-why-the-cpu-version-does-not-need-upsweepdownsweep)
    - [4. Architectural summary and its algorithmic implications](#4-architectural-summary-and-its-algorithmic-implications)
    - [5. Why we need the upsweep/downsweep pattern in CUDA](#5-why-we-need-the-upsweepdownsweep-pattern-in-cuda)
    - [6. Final takeaway](#6-final-takeaway)
- [Why JULIA NUM THREADS only for CPU](#why-julia-num-threads-only-for-cpu)
    - [1. Why GPU kernels do not depend on JULIA NUM THREADS](#1-why-gpu-kernels-do-not-depend-on-julia-num-threads)
    - [2. Why CPU parallel scan depends on JULIA NUM THREADS](#2-why-cpu-parallel-scan-depends-on-julia-num-threads)
    - [3. Why must keep these two concepts separate](#3-why-must-keep-these-two-concepts-separate)
    - [4. The two models can coexist](#4-the-two-models-can-coexist)
- [Dynamic and static shared mem with CUDA.jl](#dynamic-and-static-shared-mem-with-cudajl)
    - [1. What “static shared memory” means](#1-what-static-shared-memory-means)
    - [2. What “dynamic shared memory” means](#2-what-dynamic-shared-memory-means)
    - [3. Practical differences](#3-practical-differences)
    - [4. Why dynamic version failed](#4-why-dynamic-version-failed)
    - [5. Recommendation](#5-recommendation)
- [How warp-level scans reduce sync overhead](#how-warp-level-scans-reduce-sync-overhead)
    - [1. The fundamental observation](#1-the-fundamental-observation)
    - [2. Cost of our current Blelloch scan](#2-cost-of-our-current-blelloch-scan)
    - [3. Warp-level scan: the basic idea](#3-warp-level-scan-the-basic-idea)
    - [4. Warp-level primitives in CUDA.jl](#4-warp-level-primitives-in-cudajl)
    - [5. Step-by-step warp scan (inclusive)](#5-step-by-step-warp-scan-inclusive)
    - [6. Block-wide scan using warps (outline)](#6-block-wide-scan-using-warps-outline)
        - [Phase 1 - Warp-local scan](#phase-1---warp-local-scan)
        - [Phase 2 - Scan warp sums](#phase-2---scan-warp-sums)
        - [Phase 3 - Uniform add](#phase-3---uniform-add)
    - [7. Synchronization comparison](#7-synchronization-comparison)
    - [8. Why this works safely](#8-why-this-works-safely)
    - [9. CUDA.jl-style sketch (simplified)](#9-cudajl-style-sketch-simplified)
    - [10. Practical consequence](#10-practical-consequence)

<!-- /TOC -->

# How do we usually use threadIdx()

In a CUDA kernel (whether in CUDA C or CUDA.jl), the expression:

```julia
tid = threadIdx().x
```

returns the **x-dimension index of the currently executing thread within its block**.

This is how each GPU thread knows “who it is” so that it can compute which element of the array it should operate on.

---

## 1. What `threadIdx().x` actually is

Every CUDA kernel is launched with:

* some number of **blocks**, and
* each block contains some number of **threads**.

CUDA.jl exposes this exactly like CUDA C:

* `threadIdx().x`: thread index *inside* the current block
* `blockIdx().x`: index *of* the block
* `blockDim().x`: how many threads per block
* `gridDim().x`: how many blocks in the grid

`threadIdx().x` is a **1-based** integer in Julia.
(Important: CUDA C uses 0-based indexing; CUDA.jl uses 1-based indexing to match Julia conventions.)

So if launch a kernel with:

```julia
@cuda threads=1024 mykernel(...)
```

then inside `mykernel`:

* `threadIdx().x` ranges from **1 to 1024**

but all 1024 threads execute the function simultaneously.

---

## 2. Why we use it

If we want each thread to operate on element `i`, we typically compute:

```julia
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
```

This is the canonical GPU formula for retrieving the global “thread index” across the whole grid.

For example:

* threads per block = 1024
* block index = 3
* thread index = 17

Then:

```
global index = (3 - 1) * 1024 + 17 = 2049
```

That thread would operate on element 2049 in the array.

---

## 3. Applying this to the Blelloch scan kernel

The 1st version in [prefixsum-para-gpu-static.ipynb](./prefixsum-para-gpu-static.ipynb) uses a **single block**, so:

* `blockIdx().x == 1`
* only `threadIdx().x` matters.

Each thread uses its ID to know which element of the shared memory buffer (`sdata`) it is responsible for.

This is why the code checks:

```julia
if tid <= n
    sdata[tid] = data[tid]
end
```

The thread with `tid = 7` loads `data[7]`, thread 8 loads `data[8]`, and so on.

---

## 4. How CUDA executes this

When launching a kernel with N threads:

1. The kernel function appears to run "once", but in reality it is run simultaneously by N threads.
2. Each thread sees a different `threadIdx().x`.
3. All threads execute the same code, but on different data, using their IDs to compute their part.

This programming model is called **SIMT** (Single Instruction, Multiple Threads).

---

## 5. Small example to illustrate

```julia
function demo_kernel()
    tid = threadIdx().x
    @cuprintf("Hello from thread %d\n", tid)
end

@cuda threads=4 demo_kernel()
```

Possible output:

```
Hello from thread 1
Hello from thread 2
Hello from thread 3
Hello from thread 4
```

Four threads executed the same kernel function, each seeing a different `threadIdx().x`.

---

## 6. Summary

| Expression      | Meaning                                      |
| --------------- | -------------------------------------------- |
| `threadIdx().x` | Which thread am I inside my block? (1-based) |
| `blockIdx().x`  | Which block am I? (1-based)                  |
| `blockDim().x`  | How many threads per block?                  |
| `gridDim().x`   | How many blocks?                             |

Together these give each GPU thread a unique identity so it knows which synthetic “piece of work” to do.

# The x y z of threadIdx()

In CUDA (and therefore CUDA.jl), **`threadIdx` is not a scalar**. It is a **3-component index structure**, and `.x` selects one of its components.

This is a direct reflection of CUDA’s execution model.

---

## 1. CUDA threads are indexed in 3 dimensions

CUDA defines thread indices as a **3D vector**:

* `x`
* `y`
* `z`

This applies to:

* `threadIdx` (thread index within a block)
* `blockIdx` (block index within the grid)
* `blockDim` (block dimensions)
* `gridDim` (grid dimensions)

In CUDA.jl:

```julia
threadIdx()  :: NamedTuple { x, y, z }
```

So:

```julia
threadIdx().x
threadIdx().y
threadIdx().z
```

are simply **field accesses**, not variables.

---

## 2. Why `.x` exists even if we only use 1D threads

Even when we launch a kernel with:

```julia
@cuda threads=n
```

we are implicitly launching a **1D thread block**:

```
blockDim = (n, 1, 1)
```

So inside the kernel:

```julia
threadIdx().x ∈ [1, n]
threadIdx().y = 1
threadIdx().z = 1
```

The `.x` component is used by convention for 1D parallelism.

---

## 3. Analogy with arrays

Can think of it like this:

```julia
idx = (x = 5, y = 1, z = 1)
idx.x  # → 5
```

`threadIdx()` returns a small struct (or named tuple), and `.x` is just a field lookup.

---

## 4. Why CUDA uses x/y/z instead of i/j/k

This is historical and architectural:

* CUDA hardware natively supports **up to 3D grids**
* Common use cases:

  * 1D: vectors (use `.x`)
  * 2D: images / matrices (use `.x`, `.y`)
  * 3D: volumes (use `.x`, `.y`, `.z`)

Example 2D kernel indexing:

```julia
i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
```

CUDA.jl keeps this model intact to remain a thin, predictable layer over CUDA.

---

## 5. Important Julia-specific detail (1-based indexing)

One subtle but critical CUDA.jl detail:

* CUDA C/C++: `threadIdx.x` is **0-based**
* CUDA.jl: `threadIdx().x` is **1-based**

This is why the kernel explicitly converts:

```julia
t = tid - 1
```

Failing to do this would break the Blelloch indexing logic.

---

## Summary

* `threadIdx()` returns a **3D index object**
* `.x` selects the **x-dimension component**
* It is a **field access**, not a variable
* Even 1D kernels use `.x` by convention
* CUDA.jl uses **1-based indexing**, unlike CUDA C++


# What does syncthreads() do

`sync_threads()` is the CUDA.jl name for a **block-wide barrier synchronization**. It is the direct analogue of CUDA C/C++’s `__syncthreads()`.

Formally and practically, it does **two things simultaneously**.

---

## 1. What `sync_threads()` guarantees

When a thread executes:

```julia
sync_threads()
```

CUDA guarantees that:

1. **All threads in the same block reach the barrier before any are allowed to proceed**
2. **All writes to shared memory performed before the barrier are visible to all threads after the barrier**

In other words, it is both:

* a **control barrier**, and
* a **shared-memory fence**

This dual role is essential for correctness.

---

## 2. Scope: block-level only

A critical constraint:

* `sync_threads()` synchronizes **only threads within the same block**
* It does **not** synchronize across blocks

In our kernel, this is fine because:

* We explicitly launch **exactly one block**
* All threads participating in the scan are in that block

This is why multi-block scans require a completely different design.

---

## 3. Why it is mandatory in our Blelloch scan

Our kernel uses **shared memory**:

```julia
sdata = @cuStaticSharedMem(T, 1024)
```

Threads cooperatively update `sdata` during:

* the **upsweep** phase
* the **downsweep** phase

Each phase consists of multiple *levels*, and **each level depends on the results of the previous level**.

Example from the upsweep:

```julia
sdata[bi] += sdata[ai]
```

If some threads advanced to the next `offset` before others completed this update, the scan tree would be corrupted.

Hence the pattern:

```julia
update shared memory
sync_threads()
```

repeated at every level.

---

## 4. What would break without it

If we remove or misplace `sync_threads()`:

* Some threads would read **stale** values from `sdata`
* Others would overwrite values that have not yet been consumed
* The result would be **non-deterministic and incorrect**

This is a classic **data race**.

---

## 5. Correctness rule (very important)

Every thread in the block must either:

* reach the same `sync_threads()` call, or
* none of them do

This means:

❌ Incorrect:

```julia
if tid ≤ n
    sync_threads()
end
```

✔ Correct:

```julia
if tid ≤ n
    # work
end
sync_threads()
```

The kernel respects this rule.

The only caveat is the early return:

```julia
if tid <= n
    sdata[tid] = data[tid]
else
    return
end
```

This is safe **only because** `threads == n`.
If `threads > n`, this would deadlock the kernel.

---

## 6. Performance implications

`sync_threads()` is:

* relatively expensive
* forces all threads to wait for the slowest one

However, in tree-based scans:

* barriers are unavoidable
* the algorithmic structure demands them

More advanced scans reduce the number of barriers using:

* warp-level primitives
* cooperative groups
* hierarchical scans

---

## 7. Summary

* `sync_threads()` is a **block-wide barrier + memory fence**
* It ensures **ordering and visibility** of shared-memory operations
* It synchronizes **only within a block**
* It is essential for correctness in Blelloch scans
* All threads must reach it unconditionally


# GPU specific parallel pattern

A famous one is the Blelloch Scan, with classic application in the prefix sum calculation we do in this folder.

An on-paper explanation of the steps of Blelloch Scan: https://www.youtube.com/watch?v=mmYv3Haj6uc .

# Why is GPU parallel so different than CPU

Below is a precise, engineering-oriented explanation of **why the GPU kernel explicitly implements upsweep + downsweep**, while the CPU parallel version does not. This requires understanding the fundamental architectural differences between **GPU threads** and **CPU threads**, and why these differences directly dictate algorithm design in CUDA.

---

## 1. CPU vs GPU threads - they are fundamentally different

### 1.1 CPU threads

A CPU thread:

* is **large**, heavy-weight,
* has its own stack, registers, and scheduler state,
* can run arbitrary code with deep control flow,
* supports locks, condition variables, dynamic memory, recursion, etc.
* has **high single-thread performance** (branch prediction, large caches),
* is scheduled by the OS (milliseconds scale).

As a result:

* We only run a small number of CPU threads (8–64).
* Each thread can handle large chunks of work serially.

### 1.2 GPU “threads”

A CUDA GPU “thread”:

* is extremely **lightweight**,
* has few registers,
* belongs to a **warp** (32 threads executing in lockstep),
* is scheduled in hardware (nanoseconds scale),
* must obey SIMD/SIMT execution patterns,
* cannot block, sleep, or hold OS-like locks.

We may launch:

* **thousands** of blocks,
* **tens of thousands** of warps,
* **hundreds of thousands to millions** of threads.

Each thread only performs a tiny piece of the work.

**Conclusion:**
GPU threads are not “threads” in the CPU sense. They are vector lanes in a massive parallel processor.

This structural difference is why GPU-friendly algorithms look drastically different.

---

## 2. Why the GPU code explicitly has upsweep + downsweep

A prefix sum has a **global dependency chain**:

```
out[i] = x[1] + x[2] + ... + x[i]
```

A CPU thread can simply do this by:

* taking a contiguous chunk,
* computing its own prefix sum sequentially,
* combining partial sums from other threads.

This works because:

* Each CPU thread is fast enough to sequentially process thousands of elements.
* The number of threads is small.
* Inter-thread communication is expensive, so we **minimize communication** and **maximize per-thread work**.

But GPU threads:

* are extremely slow at per-thread sequential work,
* are extremely fast at massive synchronized parallel work,
* must avoid branchy code and long loops per thread,
* require load-balanced, SIMD-friendly execution.

**Therefore, a completely different parallel strategy is needed.**

### GPU-friendly algorithm: Blelloch scan

It is mathematically arranged to have:

* **log₂(N)** highly parallel steps,
* each step letting all threads do a small amount of work,
* synchronization only at specific tree levels,
* full SIMD utilization.

This is ideal for GPUs.

**Hence the explicit:**

1. **Upsweep (reduce) phase**
   Build partial sums in a binary tree structure.

2. **Set root = 0**
   Turn inclusive-sum tree into exclusive-sum tree.

3. **Downsweep phase**
   Propagate sums downward to produce the final prefix sum.

Every thread participates in every step.
This gives perfect parallel load-balancing on the GPU.

---

## 3. Why the CPU version does NOT need upsweep/downsweep

Because CPU threads have high per-thread compute capacity, the CPU algorithm can be:

1. Let each CPU thread compute a **local sequential prefix sum** on its chunk.
2. Compute prefix sums of chunk totals (small array of size ~#threads).
3. Add the appropriate offset to each chunk.

This algorithm takes advantage of CPU characteristics:

* expensive threads → use few of them
* cheap sequential work → do lots of scalar work per thread
* expensive cross-thread communication → minimize synchronization

This is the exact opposite of GPUs.

---

## 4. Architectural summary and its algorithmic implications

| Feature             | CPU Threads                            | GPU Threads                                 |
| ------------------- | -------------------------------------- | ------------------------------------------- |
| Count               | Few (8–64)                             | Thousands–millions                          |
| Cost per thread     | High                                   | Very low                                    |
| Scheduler           | OS (slow)                              | Hardware (fast)                             |
| Designed for        | Complex logic, branching, caching      | Dense SIMD parallelism                      |
| Per-thread speed    | High                                   | Low                                         |
| Communication       | Expensive                              | Cheap within block (shared memory + sync)   |
| Preferred algorithm | Chunking + small amount of parallelism | Highly parallel tree-structured computation |

**Result:**

* CPU prefix sum: “do big sequential chunks in each thread”
* GPU prefix sum: “break the problem into a tree and parallelize across thousands of tiny threads”

---

## 5. Why we need the upsweep/downsweep pattern in CUDA

The Blelloch tree structure is a foundational pattern used in many GPU algorithms:

* reductions
* scans
* histogram prefix
* stream compaction
* radix sort
* parallel graph algorithms
* sparse matrix prefix operations

If we understand how the binary tree structure maps to block-wide parallelism, we unlock 70% of hand-written GPU programming.

---

## 6. Final takeaway

The GPU version performs upsweep + downsweep because the GPU requires many threads to cooperate in a highly parallel way.

The CPU version does not use this algorithm because CPU threads are powerful enough to do large parts of the work sequentially and minimizing synchronization is more important.


# Why JULIA NUM THREADS only for CPU

We do not need to set `JULIA_NUM_THREADS` for the GPU version at all.

We set it only for CPU-threaded parallelism.

---

## 1. Why GPU kernels do not depend on JULIA NUM THREADS

A CUDA kernel launch (via `@cuda`) specifies its own parallelism:

```julia
@cuda threads=1024 blocks=128 mykernel(args...)
```

This directly tells the GPU:

* how many GPU blocks to launch,
* how many GPU threads per block to create.

CUDA.jl submits this configuration to the CUDA driver, and the GPU hardware creates **all the threads**. None of these threads are Julia threads. They are hardware-level GPU threads.

**Therefore:**

* GPU thread count does not depend on how many CPU threads Julia has.
* GPU kernel parallelism is controlled entirely by the CUDA launch parameters.
* GPU threads are not scheduled by Julia or by the OS.

**Conclusion:**
**GPU = many lightweight hardware threads → independent of `JULIA_NUM_THREADS`.**

---

## 2. Why CPU parallel scan depends on JULIA NUM THREADS

The CPU parallel scan uses Julia’s internal multi-threading:

```julia
using Base.Threads

@threads for tid in 1:nthreads
    ...
end
```

The number of threads used here is:

* whatever Julia started with at process launch, which is determined by:

  * environment variable `JULIA_NUM_THREADS`, or
  * command-line option `--threads`.

If we do not set it, Julia defaults to **1 thread**.

So the CPU version uses only one core unless:

```bash
export JULIA_NUM_THREADS=8
```

or:

```bash
julia --threads 8
```

In other words:

**CPU = small number of heavyweight OS threads → controlled by Julia’s threading settings.**

---

## 3. Why must keep these two concepts separate

| Feature        | CPU parallel scan   | GPU scan                       |
| -------------- | ------------------- | ------------------------------ |
| Parallel units | CPU threads (heavy) | GPU threads (massive)          |
| Controlled by  | `JULIA_NUM_THREADS` | `@cuda threads=... blocks=...` |
| Spawned by     | Julia runtime / OS  | GPU hardware scheduler         |
| Scope          | CPU cores only      | GPU multiprocessors            |
| Count          | 4–64 typical        | Thousands–millions             |

Setting `JULIA_NUM_THREADS` only affects CPU-based multithreading.
It has **zero impact** on CUDA.jl kernels.

---

## 4. The two models can coexist

We can write code where Julia CPU threads orchestrate the work and GPU threads do the work:

```julia
@threads for i in 1:Chunks
    @cuda kernel(...)
end
```

But those layers are completely separate:

* CPU threads launch GPU kernels.
* GPU threads run inside the kernels.
* They do not interfere or overlap conceptually.

Even in that case, `JULIA_NUM_THREADS` affects only the CPU side.

# Dynamic and static shared mem with CUDA.jl

This is about why our CUDA.jl kernel behaved differently on Windows.

---

## 1. What “static shared memory” means

In CUDA (C, C++, Julia, etc.), **static shared memory** is declared with a compile-time size:

```
__shared__ float buf[1024];       // CUDA C example
@cuStaticSharedMem(Float32, 1024) sdata   # CUDA.jl
```

Characteristics:

* **Size is fixed at compile time.**
* Compiler knows the exact amount, so:

  * Memory layout is predetermined.
  * PTX/SASS can fully allocate it at kernel load time.
  * No size must be passed when launching the kernel.
* It is placed in the “static” segment of shared memory for the block.
* Often gives the **best reliability and sometimes better performance** because the compiler can optimize access.

When to use:

* When we know maximum size per block.
* When constraints are simple (e.g., block always uses 1024 elements).

---

## 2. What “dynamic shared memory” means

**Dynamic shared memory** is when the kernel requests an array whose length is determined at kernel launch:

```
extern __shared__ float buf[];    // CUDA C
@cuDynamicSharedMem(Float32, sdata)   # CUDA.jl
```

And we pass size at launch:

```
@cuda shmem=N*sizeof(Float32) kernel(...)
```

Characteristics:

* Size is **decided at runtime**, not compile time.
* Allows flexible kernels: different block sizes, different workloads.
* Compiler must treat the buffer as an **opaque byte region**.
* Indexing must be careful; mistakes are common.
* Requires higher-level compiler/runtime cooperation to place memory.

In CUDA.jl, this is the tricky part:

CUDA.jl generates device code and passes a dynamic shared-memory pointer via the kernel launch configuration. On Windows, using WSL2 or certain driver versions, the dynamic shared memory lowering can hit problems because:

* CUDA.jl needs to allocate dynamic shared memory at launch.
* If the driver or compilation mode has mismatches (PTX linking, JIT mode), the pointer region may not be bound correctly.
* Indexing that is slightly off or exceeds the requested byte size often silently produces incorrect results instead of crashes.

**This is why the kernel “ran” but the results were garbage.**

---

## 3. Practical differences

| Feature                       | Static Shared Memory            | Dynamic Shared Memory                   |
| ----------------------------- | ------------------------------- | --------------------------------------- |
| Size known at compile time?   | Yes                             | No (provided when launching the kernel) |
| Performance                   | Typically slightly faster       | Slightly more overhead                  |
| Compiler optimizations        | Better                          | Limited                                 |
| Indexing errors               | Often caught during development | More likely to cause silent corruption  |
| Usage style                   | Safe, fixed                     | Flexible, generic                       |
| CUDA.jl reliability (Windows) | High                            | Sometimes problematic, driver-dependent |

---

## 4. Why dynamic version failed

Our original dynamic kernel had issues:

1. Windows/Julia GPU stack is more sensitive to:

   * Using the wrong dynamic size
   * Incorrect alignment
   * Aliasing issues inside shared memory

2. Blelloch scan requires **precise power-of-two indexing**.
   If `N != blockDim` or `blockDim != dynamic size`, we get silent output errors.

Static memory hides these issues because:

* Layout guaranteed correct.
* Array bound implicit and safe.
* No allocation mismatch.

---

## 5. Recommendation

For educational GPU-programming, especially under CUDA.jl:

* Prefer **static shared memory** until:

  * we need variable-sized blocks
  * we write kernels to be reused for multiple input sizes
* Move to **dynamic shared memory** only when scaling beyond fixed-size blocks.

Professional GPU codebases (e.g., Thrust, CUB, CUDA primitives) typically use dynamic shared memory because they need generality - but they also include highly defensive code to handle alignment and block sizing.


# How warp-level scans reduce sync overhead

A technique called **warp-level scans** reduces synchronization overhead. It is used in some libraries. We are not implementing that, but it is nice to know how it helps. Because we have seen how our pedagogical version without warp-level scans is not that fast.

This explanation quickly starts to use our Blelloch implementation as an example for illustration, since we are familiar with it.

---

## 1. The fundamental observation

A **warp** (typically 32 threads) has two crucial properties:

1. All threads in a warp execute in **lockstep** (SIMT)
2. Warp-level instructions are **implicitly synchronized**

Therefore:

> **Within a warp, no `sync_threads()` is required.**

This is the key to reducing synchronization overhead.

---

## 2. Cost of our current Blelloch scan

For a block of `n` threads, Blelloch requires:

* `log₂(n)` barriers in the upsweep
* `log₂(n)` barriers in the downsweep

Total:

```
2·log₂(n) sync_threads()
```

For `n = 1024`, this is **20 full-block barriers**, each stalling all warps.

---

## 3. Warp-level scan: the basic idea

We decompose the scan hierarchically:

1. **Each warp scans its own 32 elements**

   * No barriers
   * Use warp shuffle instructions
2. **One warp scans the per-warp sums**

   * Very small problem (≤ 32 values)
3. **Each warp adds its prefix offset**

   * Again, no barriers inside the warp

Only **two block-level synchronizations** are required.

---

## 4. Warp-level primitives in CUDA.jl

CUDA.jl exposes warp shuffles via `shfl_*` intrinsics.

The most common one for scans:

```julia
shfl_up_sync(mask, value, offset)
```

* Moves `value` from lane `laneId - offset`
* Only within the same warp
* Implicitly synchronized

---

## 5. Step-by-step warp scan (inclusive)

Within a warp:

```julia
lane = (threadIdx().x - 1) % warpSize()
x = data[tid]

for offset in (1, 2, 4, 8, 16)
    y = shfl_up_sync(0xffffffff, x, offset)
    if lane ≥ offset
        x += y
    end
end
```

Key points:

* No shared memory
* No `sync_threads()`
* All communication is register-to-register

---

## 6. Block-wide scan using warps (outline)

### Phase 1 - Warp-local scan

Each warp computes:

* `x` = prefix sum within warp
* The **last lane** writes its warp sum to shared memory

```julia
if lane == warpSize() - 1
    warp_sums[warp_id] = x
end
```

One `sync_threads()` after this.

---

### Phase 2 - Scan warp sums

Only the **first warp** participates:

```julia
if warp_id == 0
    # scan warp_sums using shuffles
end
```

Another `sync_threads()`.

---

### Phase 3 - Uniform add

Each warp adds its scanned offset:

```julia
if warp_id > 0
    x += warp_sums[warp_id - 1]
end
```

No barrier needed afterward.

---

## 7. Synchronization comparison

| Method                 | Barriers (`sync_threads`) |
| ---------------------- | ------------------------- |
| Blelloch (block-wide)  | `2·log₂(n)`               |
| Warp-hierarchical scan | **2 total**               |

For `n = 1024`:

* Blelloch: 20 barriers
* Warp-level: 2 barriers

This is a **dramatic reduction** in stall time.

---

## 8. Why this works safely

* Warp execution is **implicitly synchronized**
* Shuffle operations are **deterministic and race-free**
* Shared memory is touched only at warp boundaries
* Barriers are used only where cross-warp visibility is required

---

## 9. CUDA.jl-style sketch (simplified)

```julia
function warp_scan(x)
    lane = (threadIdx().x - 1) % warpSize()
    for offset in (1, 2, 4, 8, 16)
        y = shfl_up_sync(0xffffffff, x, offset)
        if lane ≥ offset
            x += y
        end
    end
    return x
end
```

This function contains **zero barriers** and replaces an entire Blelloch level.

---

## 10. Practical consequence

Modern high-performance scan implementations:

* Rarely use full-block Blelloch anymore
* Use warp-level scans + minimal block sync
* Scale efficiently to thousands of elements

This is how CUDA libraries such as **CUB** and **Thrust** implement scans internally.
