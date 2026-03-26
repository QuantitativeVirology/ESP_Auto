#pragma once
#include <stdint.h>

typedef enum {
    LAYER_CONV2D,
    LAYER_DEPTHWISE_CONV2D,
    LAYER_DENSE,
    LAYER_GLOBAL_AVG_POOL,
} layer_type_t;

typedef enum {
    QUANT_INT8,
    QUANT_TERNARY,
} quant_mode_t;

typedef struct {
    layer_type_t type;
    quant_mode_t quant;
    const void *weights;
    const int32_t *bias;
    float scale_pos;
    float scale_neg;
    float requant_scale;
    int8_t requant_zp;
    int in_c, out_c;
    int kernel, stride, padding;
} layer_config_t;

#define CLASS_CAT 0
#define CLASS_DOG 1

int classify_image(const int8_t *input_96x96x3);
void inference_print_memory_map(void);
