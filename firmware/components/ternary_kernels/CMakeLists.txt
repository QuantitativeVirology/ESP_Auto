idf_component_register(
    SRCS "src/ternary_conv2d_ref.c"
         "src/ternary_dense_ref.c"
         "src/int8_kernels.c"
         "src/ternary_conv2d.S"
         "src/ternary_dense.S"
    INCLUDE_DIRS "include"
)
