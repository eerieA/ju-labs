using CUDA
using Printf

CUDA.allowscalar(false)  # avoid accidental slow scalar ops

function time_circshift_vs_rotate(sx, sy; reps=100)
    println("\n" * "="^60)
    println("Size: $sx × $sy  (reps=$reps)")
    println("="^60)
    
    # Allocate device arrays
    A = CUDA.rand(Float32, sx, sy)
    B = similar(A)
    C = similar(A)
    
    # Warm-up
    for _ in 1:5
        temp = circshift(A, (1, 0))
        CUDA.synchronize()
    end
    
    # Create CUDA events for timing
    ev_start = CUDA.CuEvent()
    ev_stop  = CUDA.CuEvent()
    
    # 1) circshift (allocating version)
    CUDA.record(ev_start)
    for _ in 1:reps
        temp = circshift(A, (1, 0))
    end
    CUDA.record(ev_stop)
    CUDA.synchronize(ev_stop)
    t_ms_circ_alloc = CUDA.elapsed(ev_start, ev_stop)
    @printf("circshift (allocating):  total=%.3f ms, per-call=%.6f ms\n", 
            t_ms_circ_alloc, t_ms_circ_alloc/reps)
    
    # 2) circshift! (in-place version, but on 2D arrays it still does work)
    CUDA.record(ev_start)
    for _ in 1:reps
        circshift!(C, A, (1, 0))
    end
    CUDA.record(ev_stop)
    CUDA.synchronize(ev_stop)
    t_ms_circ_inplace = CUDA.elapsed(ev_start, ev_stop)
    @printf("circshift! (in-place):   total=%.3f ms, per-call=%.6f ms\n", 
            t_ms_circ_inplace, t_ms_circ_inplace/reps)
    
    # 3) copy! (full device copy as reference for O(n) cost)
    CUDA.record(ev_start)
    for _ in 1:reps
        copy!(B, A)
    end
    CUDA.record(ev_stop)
    CUDA.synchronize(ev_stop)
    t_ms_copy = CUDA.elapsed(ev_start, ev_stop)
    @printf("copy! (memcpy):          total=%.3f ms, per-call=%.6f ms\n", 
            t_ms_copy, t_ms_copy/reps)
    
    # 4) Pointer rotation benchmark (Vector of CuArrays)
    # Simulate what a wave equation PDE solver might do, like time rotation
    data = [CUDA.rand(Float32, sx, sy) for _ in 1:3]
    
    # Warm-up
    for _ in 1:5
        tmp = data[3]
        data[3] = data[2]
        data[2] = data[1]
        data[1] = tmp
    end
    
    CUDA.record(ev_start)
    for _ in 1:reps
        tmp = data[3]
        data[3] = data[2]
        data[2] = data[1]
        data[1] = tmp
    end
    CUDA.record(ev_stop)
    CUDA.synchronize(ev_stop)
    t_ms_rotate = CUDA.elapsed(ev_start, ev_stop)
    @printf("rotate_timelevels!:      total=%.3f ms, per-call=%.6f ms\n", 
            t_ms_rotate, t_ms_rotate/reps)
    
    # 5) circshift! on Vector of CuArrays    
    # Simulate what a wave equation PDE solver might do
    data2 = [CUDA.rand(Float32, sx, sy) for _ in 1:3]
    
    # Warm-up
    for _ in 1:5
        circshift!(data2, 1)
    end
    
    CUDA.record(ev_start)
    for _ in 1:reps
        circshift!(data2, 1)
    end
    CUDA.record(ev_stop)
    CUDA.synchronize(ev_stop)
    t_ms_circ_vec = CUDA.elapsed(ev_start, ev_stop)
    @printf("circshift! (Vector):     total=%.3f ms, per-call=%.6f ms\n", 
            t_ms_circ_vec, t_ms_circ_vec/reps)
    
    println("\nSpeedups relative to rotate_timelevels!:")
    @printf("  circshift! (Vector) is %.1fx slower\n", t_ms_circ_vec/t_ms_rotate)
    @printf("  circshift (allocating on array) is %.1fx slower\n", t_ms_circ_alloc/t_ms_rotate)
    
    return nothing
end

# Sweep sizes
println("GPU detected: ", CUDA.device())
println("CUDA capability: ", CUDA.capability(CUDA.device()))

sizes = [512, 1024, 2048]  # Extend if memory allows
for s in sizes
    reps = s <= 1024 ? 1000 : s <= 2048 ? 500 : 100
    time_circshift_vs_rotate(s, s; reps=reps)
end