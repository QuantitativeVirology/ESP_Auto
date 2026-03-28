idf_component_register(
    SRCS "src/ternary_conv2d_ref.c"
         "src/ternary_dense_ref.c"
         "src/ternary_conv2d_simd.c"
         "src/ternary_dense_simd.c"
         "src/int8_kernels.c"
    INCLUDE_DIRS "include"
)
