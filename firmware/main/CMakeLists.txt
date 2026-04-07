idf_component_register(
    SRCS "main.c" "camera.c" "inference.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES driver esp_timer ternary_kernels
)

# Enable embedded test images for RUN_BENCHMARK command
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_images.h")
    target_compile_definitions(${COMPONENT_LIB} PRIVATE INCLUDE_TEST_IMAGES)
endif()
