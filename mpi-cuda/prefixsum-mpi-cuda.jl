using MPI
using CUDA

# -------------------------------
# GPU kernel: add scalar offset
# -------------------------------
function add_offset_kernel!(y, offset)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] += offset
    end
    return
end

# -------------------------------
# Simple GPU scan using cumsum
# -------------------------------
function gpu_scan(x::CuArray{T}) where {T}
    return cumsum(x)   # inclusive scan on GPU
end

# -------------------------------
# Main program
# -------------------------------
MPI.Init()
try
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    # --------------------------------
    # Problem size
    # --------------------------------
    N_global = 1024        # total length
    @assert N_global % size == 0 "N_global must be divisible by number of MPI ranks"

    N_local = N_global ÷ size

    # --------------------------------
    # Initialize local data
    # --------------------------------
    x_local = fill(Float32(rank + 1), N_local)  # deterministic values for testing

    # --------------------------------
    # Move data to GPU
    # --------------------------------
    d_x = CuArray(x_local)

    # --------------------------------
    # 1. Local scan on GPU
    # --------------------------------
    d_scan = gpu_scan(d_x)

    # --------------------------------
    # 2. Local total (GPU reduction -> CPU scalar)
    # --------------------------------
    local_total = sum(d_x)      # convert GPU scalar to CPU Float32

    # --------------------------------
    # 3. MPI exclusive scan of totals
    # --------------------------------
    offset = MPI.Exscan(local_total, +, comm)

    # MPI.Exscan returns `nothing` on rank 0
    if rank == 0
        offset = zero(Float32)
    end

    # --------------------------------
    # 4. Apply offset on GPU
    # --------------------------------
    threads = 256
    blocks = cld(N_local, threads)
    @cuda threads=threads blocks=blocks add_offset_kernel!(d_scan, offset)

    # --------------------------------
    # Copy result back to CPU
    # --------------------------------
    y_local = Array(d_scan)

    # --------------------------------
    # Print for verification (rank by rank)
    # --------------------------------
    MPI.Barrier(comm)
    for r in 0:size-1
        MPI.Barrier(comm)
        if rank == r
            println("Rank $rank result:")
            println(y_local)
        end
    end

finally
    MPI.Finalize()
end
