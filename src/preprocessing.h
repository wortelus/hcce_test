#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>

namespace hcce {

    // Preprocessing vstupního obrazu před YOLO i HccePose inference
    class Preprocessing {
    public:
        // ─── Pro YOLO ──────────────────────────────────────────────────────────────
        // Resize na max straně 640, padding na násobek 32
        // Vrací: upravený obraz a scaling faktor + offset (pro zpětnou transformaci bbox)
        struct YoloPrepResult {
            cv::Mat   img;        // upravený obraz [H_pad, W_pad, 3] uint8
            float     scale;      // faktor zmenšení (orig / new)
            int       pad_left;
            int       pad_top;
        };
        static YoloPrepResult prepareForYolo(const cv::Mat& img_bgr,
                                              int max_side = 640,
                                              int pad_multiple = 32);

        // ─── Pro HccePose ──────────────────────────────────────────────────────────
        // Vstup: crop 256×256 uint8 RGB
        // Výstup: float32 CHW tensor [3, 256, 256] normalizovaný ImageNet
        static std::vector<float> prepareForHccePose(const cv::Mat& crop_rgb);

        // ─── Pomocné ───────────────────────────────────────────────────────────────
        // BGR → RGB + resize + konverze na float
        static cv::Mat bgrToRgbFloat(const cv::Mat& img_bgr);

        static std::vector<float> floatMatToCHW(const cv::Mat& img_float);

    private:
        // ImageNet mean/std pro normalizaci
        static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
        static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};
    };

} // namespace hcce