/*
 * ternary_dense_simd — Xtensa PIE SIMD ternary dense kernel
 *
 * TODO: implement SIMD version (Phase 4)
 * For now, this is a thin wrapper calling the C reference.
 */

    .text
    .align 4
    .global ternary_dense_simd
    .type   ternary_dense_simd, @function

ternary_dense_simd:
    j ternary_dense_ref

    .size ternary_dense_simd, . - ternary_dense_simd
