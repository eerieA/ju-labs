<!-- TOC -->

- [Associative and commutative operations](#associative-and-commutative-operations)
    - [Associative Operations](#associative-operations)
    - [Commutative Operations](#commutative-operations)
    - [Relationship Between the Two](#relationship-between-the-two)
    - [Practical Relevance (Parallel Computing Context)](#practical-relevance-parallel-computing-context)
- [Conceptual hybrid MPI + CUDA prefix sum](#conceptual-hybrid-mpi--cuda-prefix-sum)
    - [1. Problem statement (global scan)](#1-problem-statement-global-scan)
    - [2. Data decomposition (MPI level)](#2-data-decomposition-mpi-level)
    - [3. Step 1: Local prefix sum on each GPU (CUDA)](#3-step-1-local-prefix-sum-on-each-gpu-cuda)
    - [4. Step 2: Global scan of block totals (MPI level)](#4-step-2-global-scan-of-block-totals-mpi-level)
    - [5. Step 3: Offset local scans (CUDA again)](#5-step-3-offset-local-scans-cuda-again)
    - [6. Final result](#6-final-result)
    - [7. Where CUDA-aware MPI fits](#7-where-cuda-aware-mpi-fits)
    - [8. Minimal pseudocode (Julia-flavored)](#8-minimal-pseudocode-julia-flavored)
    - [9. Key conceptual insight](#9-key-conceptual-insight)
- [Derivation of a MPI Exscan tree](#derivation-of-a-mpi-exscan-tree)
    - [1. What MPI Exscan must compute](#1-what-mpi-exscan-must-compute)
    - [2. Relation to Blelloch scan](#2-relation-to-blelloch-scan)
    - [3. Canonical implementation: binomial tree Exscan](#3-canonical-implementation-binomial-tree-exscan)
    - [4. Rank notation and assumptions](#4-rank-notation-and-assumptions)
    - [5. Communication rounds (bitwise construction)](#5-communication-rounds-bitwise-construction)
    - [6. Concrete example: P = 8](#6-concrete-example-p--8)
        - [Step d = 0 (distance = 1)](#step-d--0-distance--1)
        - [Step d = 1 (distance = 2)](#step-d--1-distance--2)
        - [Step d = 2 (distance = 4)](#step-d--2-distance--4)
    - [7. Why this computes the *exclusive* scan](#7-why-this-computes-the-exclusive-scan)
    - [8. Relationship to Blelloch phases](#8-relationship-to-blelloch-phases)
    - [9. Why rank 0 is special](#9-why-rank-0-is-special)
    - [10. Pseudocode (conceptual, not MPI API)](#10-pseudocode-conceptual-not-mpi-api)
    - [11. Performance characteristics](#11-performance-characteristics)
    - [Caveat and takeaway](#caveat-and-takeaway)
- [Why is the down-sweep called exclusive phase](#why-is-the-down-sweep-called-exclusive-phase)
    - [1. Inclusive vs exclusive prefix sums](#1-inclusive-vs-exclusive-prefix-sums)
        - [Inclusive scan](#inclusive-scan)
        - [Exclusive scan](#exclusive-scan)
    - [2. What the up-sweep actually computes](#2-what-the-up-sweep-actually-computes)
    - [3. The key transformation in the down-sweep](#3-the-key-transformation-in-the-down-sweep)
    - [4. The “exclusion” happens explicitly](#4-the-exclusion-happens-explicitly)
    - [5. Why this cannot be done in the up-sweep](#5-why-this-cannot-be-done-in-the-up-sweep)
    - [6. MPIExscan perspective](#6-mpiexscan-perspective)
    - [7. Summary](#7-summary)

<!-- /TOC -->

# Associative and commutative operations

## Associative Operations

An operation (\circ) is **associative** if the *grouping* of operands does not affect the result.

**Formal definition**

For all (a, b, c) in a set:
$$
(a \circ b) \circ c = a \circ (b \circ c)
$$

**Key idea**
We may change parentheses without changing the outcome.

**Examples (associative)**

* Addition: ((a + b) + c = a + (b + c))
* Multiplication: ((ab)c = a(bc))
* Maximum / Minimum: (\max(\max(a,b),c) = \max(a,\max(b,c)))
* Bitwise AND / OR

**Non-examples (not associative)**

* Subtraction: ((a - b) - c \neq a - (b - c))
* Division: ((a / b) / c \neq a / (b / c))

**Why associativity matters**
Associativity enables:

* Tree-based reductions (e.g., prefix sums, reductions)
* Reordering of computation across processors
* Efficient parallel algorithms (MPI, CUDA, OpenMP)

---

## Commutative Operations

An operation (\circ) is **commutative** if the *order* of operands does not affect the result.

**Formal definition**

For all (a, b):
$$
a \circ b = b \circ a
$$

**Key idea**
We may swap operands freely.

**Examples (commutative)**

* Addition: (a + b = b + a)
* Multiplication: (ab = ba)
* Maximum / Minimum
* Bitwise AND / OR

**Non-examples (not commutative)**

* Subtraction: (a - b \neq b - a)
* Division: (a / b \neq b / a)
* Matrix multiplication (in general)

---

## Relationship Between the Two

They are **independent**. For example:

| Operation             | Associative? | Commutative? |
| --------------------- | ------------ | ------------ |
| Addition              | Yes          | Yes          |
| Multiplication        | Yes          | Yes          |
| Subtraction           | No           | No           |
| Division              | No           | No           |
| Matrix multiplication | Yes          | No           |

An operation may be:

* Associative but not commutative (e.g., matrix multiplication)
* Commutative but not associative (rare but possible in abstract algebra)
* Both (most reductions)
* Neither

---

## Practical Relevance (Parallel Computing Context)

* **Associativity is essential** for parallel prefix sums and reductions
  (it allows regrouping across threads/ranks).
* **Commutativity is optional but helpful**
  (it allows arbitrary reordering and load balancing).
* MPI collective operations typically require *associativity*; some also assume commutativity for optimization.


# Conceptual hybrid MPI + CUDA prefix sum

## 1. Problem statement (global scan)

We want to compute a prefix sum over a very large array:

$$
y_i = \sum_{k=0}^{i} x_k,\quad i = 0,\dots,N-1
$$

The array is **distributed across MPI ranks**, and each rank may use a **GPU** to accelerate local computation.

---

## 2. Data decomposition (MPI level)

Assume:

* `P` MPI ranks
* Global array size `N`
* Each rank owns a contiguous chunk of size `N / P`

```
Global array:
[x0 x1 x2 ... xN-1]

Rank 0 owns: [x0 ... x_{n-1}]
Rank 1 owns: [x_n ... x_{2n-1}]
Rank 2 owns: [x_{2n} ... x_{3n-1}]
...
```

This **ownership split** is the fundamental MPI concept.

---

## 3. Step 1: Local prefix sum on each GPU (CUDA)

On **each rank**, independently:

1. Copy the local chunk to the GPU
2. Run a CUDA Blelloch scan kernel
3. Obtain:

   * `local_scan[i]`
   * `local_total = sum(local_chunk)`

This step is **identical to what we already implemented**, except it runs on only a *subset* of the data.

```
Rank k GPU computes:

input:   [a0 a1 a2 a3]
scan:    [a0 a0+a1 a0+a1+a2 a0+a1+a2+a3]
total:   T_k = a0+a1+a2+a3
```

No MPI involved yet.

---

## 4. Step 2: Global scan of block totals (MPI level)

Now each rank has **one scalar**:

```
T_0, T_1, T_2, ..., T_{P-1}
```

We need the **prefix sum of these totals**, because:

* Rank 0’s data needs no offset
* Rank 1’s data must be offset by `T_0`
* Rank 2’s data must be offset by `T_0 + T_1`
* etc.

This is a classic MPI collective:

```c
MPI_Exscan(T_k, offset_k, MPI_SUM)
```

Result:

```
offset_0 = 0
offset_1 = T_0
offset_2 = T_0 + T_1
...
```

This is logically the **same tree structure** as Blelloch, but across *processes*, not threads.

---

## 5. Step 3: Offset local scans (CUDA again)

Each rank now launches a simple CUDA kernel:

```text
local_scan[i] += offset_k
```

This is embarrassingly parallel and very fast.

---

## 6. Final result

The global prefix sum is now correct across **all ranks and all GPUs**.

The algorithm is:

1. **CUDA**: local scan
2. **MPI**: scan of block totals
3. **CUDA**: add offsets

This pattern appears everywhere in HPC.

---

## 7. Where CUDA-aware MPI fits

If an MPI implementation is CUDA-aware:

* `T_k` can live in device memory
* `MPI_Exscan` can operate directly on GPU buffers
* No host staging required

Otherwise:

* Copy `T_k` to host
* Run `MPI_Exscan`
* Copy `offset_k` back to device

Functionally identical, slower.

---

## 8. Minimal pseudocode (Julia-flavored)

```julia
using MPI, CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

\# Local data
x_local = rand(Float32, N ÷ size)

\# Move to GPU
d_x = CuArray(x_local)

\# 1. Local scan on GPU
d_scan = blelloch_scan!(d_x)

\# 2. Local total
T_local = sum(d_x)  # GPU reduction

\# 3. MPI exclusive scan
offset = MPI.Exscan(T_local, +, comm)

\# Rank 0 gets no offset
if rank == 0
    offset = 0.0f0
end

\# 4. Apply offset on GPU
@cuda add_offset!(d_scan, offset)

MPI.Finalize()
```

---

## 9. Key conceptual insight

> **CUDA solves the “many operations on one chunk” problem.
> MPI solves the “many chunks across the system” problem.**

The prefix sum is a perfect example because:

* It exposes both **fine-grained** and **coarse-grained** parallelism
* The algorithmic structure repeats across abstraction levels


# Derivation of a MPI Exscan tree

Expressed in the same algorithmic language we used when reasoning about Blelloch scans.

---

## 1. What MPI Exscan must compute

Given ranks `0 … P−1` and local values `T_k`, MPI_Exscan returns:

[
\text{offset}*k = \sum*{i=0}^{k-1} T_i
]

with the convention that:

* Rank 0 receives **undefined** (often treated as zero by user code)

MPI **does not mandate a specific tree**, only correctness and associativity of the operator.

However, *all high-performance implementations use tree-based algorithms*.

---

## 2. Relation to Blelloch scan

Our CUDA Blelloch scan consists of:

1. **Up-sweep (reduce)**: build partial sums in a tree
2. **Down-sweep (propagate)**: distribute prefix values

MPI_Exscan is conceptually the **distributed-memory analog** of this pattern, but:

* Nodes are **processes**
* Communication is **explicit**
* Memory is **not shared**

---

## 3. Canonical implementation: binomial tree Exscan

The most common implementation strategy is a **binomial tree**, because:

* O(log P) steps
* Minimal messages
* Natural mapping to rank bits

We now derive it.

---

## 4. Rank notation and assumptions

Let:

* `rank` be the MPI rank
* `P` be the number of ranks
* `T[rank]` be the local contribution
* `recv` be the partial prefix sum accumulated so far

Initialize:

```
recv = 0
temp = T[rank]
```

---

## 5. Communication rounds (bitwise construction)

For step `d = 0, 1, 2, ..., ⌈log₂ P⌉ − 1`:

Define:

```
partner = rank XOR (1 << d)
```

Only ranks where:

```
rank & (1 << d) != 0
```

receive contributions from lower ranks.

---

## 6. Concrete example: P = 8

Ranks: 0 1 2 3 4 5 6 7
Binary:

```
0: 000
1: 001
2: 010
3: 011
4: 100
5: 101
6: 110
7: 111
```

---

### Step d = 0 (distance = 1)

Pairs:

```
(0 → 1), (2 → 3), (4 → 5), (6 → 7)
```

* Rank 1 receives T₀
* Rank 3 receives T₂
* Rank 5 receives T₄
* Rank 7 receives T₆

Update:

```
recv += incoming
temp += incoming
```

---

### Step d = 1 (distance = 2)

Pairs:

```
(0 → 2), (1 → 3), (4 → 6), (5 → 7)
```

* Rank 2 receives T₀ + T₁
* Rank 3 receives T₀ + T₁
* Rank 6 receives T₄ + T₅
* Rank 7 receives T₄ + T₅

---

### Step d = 2 (distance = 4)

Pairs:

```
(0 → 4), (1 → 5), (2 → 6), (3 → 7)
```

* Rank 4 receives T₀ + T₁ + T₂ + T₃
* Rank 5 receives same
* Rank 6 receives same
* Rank 7 receives same

---

## 7. Why this computes the *exclusive* scan

At each step:

* A rank only **receives from lower-ranked processes**
* Contributions always represent **contiguous rank ranges**
* No rank ever includes its own `T_k` in `recv`

Thus:

```
recv(rank k) = sum_{i < k} T_i
```

This is the exclusive property.

---

## 8. Relationship to Blelloch phases

| Blelloch (CUDA) | MPI Exscan                           |
| --------------- | ------------------------------------ |
| Threads         | Processes                            |
| Shared memory   | Message passing                      |
| Up-sweep        | Accumulation of `temp`               |
| Down-sweep      | Controlled propagation via rank bits |
| Barrier         | Point-to-point sync                  |

MPI fuses both phases into a **single logarithmic protocol**, because memory is distributed.

---

## 9. Why rank 0 is special

Rank 0:

* Never satisfies `rank & (1 << d) != 0`
* Never receives data
* Has no defined output

This mirrors Blelloch, where the root is initialized to identity.

---

## 10. Pseudocode (conceptual, not MPI API)

```text
recv = 0
temp = T[rank]

for d = 0 to log2(P)-1:
    partner = rank ^ (1 << d)

    if rank & (1 << d):
        receive x from partner
        recv += x
        temp += x
    else:
        send temp to partner
```

MPI implementations replace this with nonblocking calls, pipelining, and topology-aware routing.

---

## 11. Performance characteristics

* Messages: O(log P)
* Latency-dominated
* Payload: single scalar (or small vector)
* Network-optimal

This is why MPI_Exscan is cheap compared to the GPU scan.

---

## Caveat and takeaway

MPI is free to:

* Use recursive doubling
* Use k-ary trees
* Use hardware collectives
* Use topology-aware algorithms

But **all correct implementations must be equivalent to a prefix tree over ranks**.

**Key takeaway**

MPI_Exscan is not magic, it is a distributed prefix tree where **rank order replaces thread index**, and **message passing replaces shared memory**.

Once we see that, the CUDA ↔ MPI analogy becomes exact.


# Why is the down-sweep called exclusive phase

The term **“exclusive phase”** is not arbitrary; it follows directly from *what information is deliberately removed and redistributed during the down-sweep*.

---

## 1. Inclusive vs exclusive prefix sums

Given input:
$$
x = [x_0, x_1, x_2, x_3]
$$

### Inclusive scan

$$
[ x_0,; x_0+x_1,; x_0+x_1+x_2,; x_0+x_1+x_2+x_3 ]
$$

Each position **includes its own element**.

### Exclusive scan

$$
[ 0,; x_0,; x_0+x_1,; x_0+x_1+x_2 ]
$$

Each position contains the sum of **strictly earlier elements only**.

MPI_Exscan explicitly computes the second form.

---

## 2. What the up-sweep actually computes

After the up-sweep (reduce phase):

* Each internal node holds the **inclusive sum of its subtree**
* The root holds the **total sum**
* No rank yet has an exclusive prefix

Critically:

> At this point, every partial sum **includes the contribution of the node itself and everything to its left inside the subtree**.

So the data is still **inclusive**.

---

## 3. The key transformation in the down-sweep

The down-sweep does one essential thing:

> **It replaces each node’s value with the sum of all values strictly to its left in the global ordering.**

This is achieved by:

1. Setting the root’s value to the identity (0)
2. Propagating prefixes downward
3. Shifting accumulated sums *rightward* in the rank ordering

That is why the operation is called **exclusive**:

* Each rank *excludes its own original value*
* Each rank receives only what came before it

---

## 4. The “exclusion” happens explicitly

Look at the critical swap step (Blelloch-style):

Let:

* `r` = parent
* `l` = left child
* Before:

  * `l` holds sum of left subtree
  * `r` holds sum of left + right subtrees
* Operation:

  ```
  temp = l
  l = r
  r = r + temp
  ```

Interpretation:

* Left child gets the **prefix so far**
* Right child gets prefix + left subtree
* Neither node keeps its own contribution

This explicitly **removes self-contribution** from the value propagated downward.

---

## 5. Why this cannot be done in the up-sweep

The up-sweep:

* Combines values
* Moves information *upward*
* Has no notion of “everything before me”

Exclusive semantics require:

* Global ordering
* Knowledge of all preceding contributions

That information only exists **after the reduction tree is built**, which is why a second phase is necessary.

---

## 6. MPI_Exscan perspective

MPI defines:

$$
\text{recv}*k = \sum*{i=0}^{k-1} x_i
$$

The MPI implementation:

* First builds partial reductions (up-sweep)
* Then redistributes prefixes (down-sweep)

The **down-sweep is the moment exclusivity is enforced**.

---

## 7. Summary

| Domain                 | Name           | Meaning                        |
| ---------------------- | -------------- | ------------------------------ |
| Parallel algorithms    | Exclusive scan | Excludes self                  |
| CUDA Blelloch          | Down-sweep     | Converts inclusive → exclusive |
| MPI                    | Exscan         | Exclusive prefix across ranks  |
| Functional programming | Scanl          | Left-exclusive accumulation    |

The term is consistent across disciplines.

The down-sweep is called the exclusive phase because it is the phase in which each participant’s own contribution is intentionally removed from its result and replaced by the sum of all earlier contributions.

Without the down-sweep, we only have reductions.
With it, we obtain ordered, exclusive prefixes.

This distinction is fundamental to why scans are more powerful than reductions in parallel algorithms.