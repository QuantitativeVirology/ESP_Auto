# Research Direction: Ternary SIMD Kernel Optimization

## Objective
Minimize cycle count of ternary_conv2d.S for fixed model architecture.

## Constraints (HARD, never violate)
- Output must exactly match ternary_conv2d_ref.c (bit-exact)
- Must not exceed IRAM budget (32 KB)
- Must maintain 16-byte alignment on all memory accesses
- Must use valid Xtensa PIE SIMD instructions only

## Optimization levers
- Loop unrolling factor
- Register allocation (q0-q7 assignment)
- Software pipelining (overlap load/compute)
- Weight expansion strategy (on-the-fly vs pre-expanded)
- Dual-issue instruction scheduling

## Metric
Cycle count for Conv2d(C_in=32, C_out=64, K=3, H=48, W=48)
Correctness gate: Must pass test_kernels.c (TESTS PASSED)
