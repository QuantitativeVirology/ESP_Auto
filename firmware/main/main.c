#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "camera.h"
#include "inference.h"
#include "ternary_ops.h"

static const char *TAG = "main";

// Signal pins — avoid camera GPIOs (4-18, 43, 44)
#define PIN_DOG   GPIO_NUM_45
#define PIN_CAT   GPIO_NUM_46
#define PIN_FRAME GPIO_NUM_47

static void gpio_init_pins(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << PIN_DOG) | (1ULL << PIN_CAT) | (1ULL << PIN_FRAME),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
    gpio_set_level(PIN_DOG, 0);
    gpio_set_level(PIN_CAT, 0);
    gpio_set_level(PIN_FRAME, 0);
}

static void signal_result(int result)
{
    gpio_set_level(PIN_DOG, result == CLASS_DOG);
    gpio_set_level(PIN_CAT, result == CLASS_CAT);
}

static void clear_result(void)
{
    gpio_set_level(PIN_DOG, 0);
    gpio_set_level(PIN_CAT, 0);
}

// ---------------------------------------------------------------------------
// Benchmark mode: run inference on embedded test images
// ---------------------------------------------------------------------------

#ifdef INCLUDE_TEST_IMAGES
#include "test_images.h"

static void run_benchmark(void)
{
    ESP_LOGI(TAG, "Running benchmark on %d test images...", NUM_TEST_IMAGES);
    int correct = 0;
    uint32_t total_us = 0;

    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        // Convert uint8 [0,255] to int8 [-128,127]
        static int8_t input_buf[96 * 96 * 3] __attribute__((aligned(16)));
        for (int j = 0; j < 96 * 96 * 3; j++) {
            input_buf[j] = (int8_t)(test_images[i][j] - 128);
        }

        gpio_set_level(PIN_FRAME, 1);
        int64_t start = esp_timer_get_time();

        int result = classify_image(input_buf);

        int64_t end = esp_timer_get_time();
        gpio_set_level(PIN_FRAME, 0);

        uint32_t latency_us = (uint32_t)(end - start);
        total_us += latency_us;

        if (result == test_labels[i]) {
            correct++;
        }

        signal_result(result);
        vTaskDelay(pdMS_TO_TICKS(10));
        clear_result();
    }

    float accuracy = (float)correct / NUM_TEST_IMAGES;
    uint32_t avg_latency = total_us / NUM_TEST_IMAGES;
    size_t sram_free = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    printf("METRIC latency_us=%u accuracy=%.4f sram_free=%u\n",
           avg_latency, accuracy, (unsigned)sram_free);
}
#endif

// ---------------------------------------------------------------------------
// Live mode: continuous camera capture + inference
// ---------------------------------------------------------------------------

static void inference_task(void *arg)
{
    static uint8_t rgb_buf[96 * 96 * 3] __attribute__((aligned(16)));
    static int8_t input_buf[96 * 96 * 3] __attribute__((aligned(16)));

    while (1) {
        if (camera_capture_96x96(rgb_buf) != ESP_OK) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // uint8 [0,255] -> int8 [-128,127]
        for (int i = 0; i < 96 * 96 * 3; i++) {
            input_buf[i] = (int8_t)(rgb_buf[i] - 128);
        }

        gpio_set_level(PIN_FRAME, 1);
        int64_t start = esp_timer_get_time();

        int result = classify_image(input_buf);

        int64_t end = esp_timer_get_time();
        gpio_set_level(PIN_FRAME, 0);

        uint32_t latency_us = (uint32_t)(end - start);
        float fps = 1000000.0f / latency_us;

        printf("%s  latency=%ums  fps=%.1f\n",
               result == CLASS_DOG ? "DOG" : "CAT",
               (unsigned)(latency_us / 1000), fps);

        signal_result(result);
        vTaskDelay(pdMS_TO_TICKS(10));
        clear_result();
    }
}

// ---------------------------------------------------------------------------
// UART command processing
// ---------------------------------------------------------------------------

static void uart_cmd_task(void *arg)
{
    char line[64];

    printf("READY\n");

    while (1) {
        if (fgets(line, sizeof(line), stdin)) {
            // Strip newline
            line[strcspn(line, "\r\n")] = 0;

#ifdef INCLUDE_TEST_IMAGES
            if (strcmp(line, "RUN_BENCHMARK") == 0) {
                run_benchmark();
                continue;
            }
#endif
            if (strcmp(line, "RUN_TESTS") == 0) {
                run_kernel_tests();
                continue;
            }
            if (strcmp(line, "MEMORY") == 0) {
                inference_print_memory_map();
                continue;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void app_main(void)
{
    ESP_LOGI(TAG, "ESP32-S3 Ternary Vision Inference");

    gpio_init_pins();
    inference_init();
    inference_print_memory_map();

    esp_err_t cam_err = camera_init();
    if (cam_err != ESP_OK) {
        ESP_LOGW(TAG, "Camera init failed — benchmark mode only");
    } else {
        // Start inference on Core 1
        xTaskCreatePinnedToCore(inference_task, "infer", 8192, NULL,
                                configMAX_PRIORITIES - 1, NULL, 1);
    }

    // UART command handler on Core 0
    xTaskCreatePinnedToCore(uart_cmd_task, "uart_cmd", 4096, NULL, 5, NULL, 0);
}
