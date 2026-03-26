#include "camera.h"
#include "esp_camera.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "camera";

// OV2640 pin mapping for ESP32-S3-WROOM-CAM
#define CAM_PIN_PWDN  -1
#define CAM_PIN_RESET -1
#define CAM_PIN_XCLK   10
#define CAM_PIN_SIOD   40
#define CAM_PIN_SIOC   39
#define CAM_PIN_D7     48
#define CAM_PIN_D6     11
#define CAM_PIN_D5     12
#define CAM_PIN_D4     14
#define CAM_PIN_D3     16
#define CAM_PIN_D2     18
#define CAM_PIN_D1     17
#define CAM_PIN_D0     15
#define CAM_PIN_VSYNC  38
#define CAM_PIN_HREF   47
#define CAM_PIN_PCLK   13

esp_err_t camera_init(void)
{
    camera_config_t config = {
        .pin_pwdn     = CAM_PIN_PWDN,
        .pin_reset    = CAM_PIN_RESET,
        .pin_xclk     = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7       = CAM_PIN_D7,
        .pin_d6       = CAM_PIN_D6,
        .pin_d5       = CAM_PIN_D5,
        .pin_d4       = CAM_PIN_D4,
        .pin_d3       = CAM_PIN_D3,
        .pin_d2       = CAM_PIN_D2,
        .pin_d1       = CAM_PIN_D1,
        .pin_d0       = CAM_PIN_D0,
        .pin_vsync    = CAM_PIN_VSYNC,
        .pin_href     = CAM_PIN_HREF,
        .pin_pclk     = CAM_PIN_PCLK,

        .xclk_freq_hz = 20000000,
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_RGB565,
        .frame_size   = FRAMESIZE_QVGA,  // 320x240
        .jpeg_quality = 12,
        .fb_count     = 2,
        .fb_location  = CAMERA_FB_IN_PSRAM,
        .grab_mode    = CAMERA_GRAB_LATEST,
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return err;
    }
    ESP_LOGI(TAG, "Camera initialized (QVGA RGB565)");
    return ESP_OK;
}

static void bilinear_downscale_rgb565_to_rgb888(
    const uint16_t *src, int src_w, int src_h,
    uint8_t *dst, int dst_w, int dst_h)
{
    for (int dy = 0; dy < dst_h; dy++) {
        float sy = (float)dy * src_h / dst_h;
        int y0 = (int)sy;
        int y1 = y0 + 1 < src_h ? y0 + 1 : y0;
        float fy = sy - y0;

        for (int dx = 0; dx < dst_w; dx++) {
            float sx = (float)dx * src_w / dst_w;
            int x0 = (int)sx;
            int x1 = x0 + 1 < src_w ? x0 + 1 : x0;
            float fx = sx - x0;

            // Sample 4 RGB565 pixels
            uint16_t p00 = src[y0 * src_w + x0];
            uint16_t p01 = src[y0 * src_w + x1];
            uint16_t p10 = src[y1 * src_w + x0];
            uint16_t p11 = src[y1 * src_w + x1];

            // Extract RGB components (5-6-5 -> 8-8-8)
            #define R565(p) (((p) >> 8) & 0xF8)
            #define G565(p) (((p) >> 3) & 0xFC)
            #define B565(p) (((p) << 3) & 0xF8)

            float r = R565(p00) * (1-fx) * (1-fy) + R565(p01) * fx * (1-fy)
                    + R565(p10) * (1-fx) * fy     + R565(p11) * fx * fy;
            float g = G565(p00) * (1-fx) * (1-fy) + G565(p01) * fx * (1-fy)
                    + G565(p10) * (1-fx) * fy     + G565(p11) * fx * fy;
            float b = B565(p00) * (1-fx) * (1-fy) + B565(p01) * fx * (1-fy)
                    + B565(p10) * (1-fx) * fy     + B565(p11) * fx * fy;

            int idx = (dy * dst_w + dx) * 3;
            dst[idx + 0] = (uint8_t)(r + 0.5f);
            dst[idx + 1] = (uint8_t)(g + 0.5f);
            dst[idx + 2] = (uint8_t)(b + 0.5f);

            #undef R565
            #undef G565
            #undef B565
        }
    }
}

esp_err_t camera_capture_96x96(uint8_t *out_rgb888)
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Camera capture failed");
        return ESP_FAIL;
    }

    bilinear_downscale_rgb565_to_rgb888(
        (const uint16_t *)fb->buf, fb->width, fb->height,
        out_rgb888, 96, 96
    );

    esp_camera_fb_return(fb);
    return ESP_OK;
}
