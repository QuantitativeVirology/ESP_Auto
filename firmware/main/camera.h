#pragma once
#include "esp_err.h"
#include <stdint.h>

esp_err_t camera_init(void);
esp_err_t camera_capture_96x96(uint8_t *out_rgb888);
