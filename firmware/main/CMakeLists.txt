idf_component_register(
    SRCS "main.c" "camera.c" "inference.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES driver esp_timer ternary_kernels
)
