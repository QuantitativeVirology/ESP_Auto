/*
 * ternary_conv2d_simd — Xtensa PIE SIMD ternary convolution kernel
 *
 * Strategy: expand 2-bit ternary weights to INT8 {-1, 0, +1},
 *           then use EE.VMULAS.S8.ACCX for 16-wide MAC.
 *
 * TODO: implement SIMD version (Phase 4)
 * For now, this is a thin wrapper calling the C reference.
 */

    .text
    .align 4
    .global ternary_conv2d_simd
    .type   ternary_conv2d_simd, @function

ternary_conv2d_simd:
    /* Fall through to C reference until SIMD is implemented */
    j ternary_conv2d_ref

    .size ternary_conv2d_simd, . - ternary_conv2d_simd
