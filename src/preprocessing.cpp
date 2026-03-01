#include "preprocessing.h"
#include <stdexcept>

namespace hcce {
    constexpr float Preprocessing::MEAN[3];
    constexpr float Preprocessing::STD[3];

    // ─── YOLO preprocessing ────────────────────────────────────────────────────────
    Preprocessing::YoloPrepResult Preprocessing::prepareForYolo(
        const cv::Mat &img_bgr, int max_side, int pad_multiple) {
        YoloPrepResult res;
        int H = img_bgr.rows, W = img_bgr.cols;

        // Python: ratio_ = max(height/640, width/640), resize jen pokud ratio_ > 1
        float scale = std::max((float) H / max_side, (float) W / max_side);

        cv::Mat resized;
        if (scale > 1.0f) {
            int H_new = (int) (H / scale);
            int W_new = (int) (W / scale);
            cv::resize(img_bgr, resized, cv::Size(W_new, H_new), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img_bgr; // obraz je menší než 640 - neresize
            scale = 1.0f;
        }

        // Padding na násobek pad_multiple (32)
        int H_new = resized.rows, W_new = resized.cols;
        int H_pad = (H_new % pad_multiple == 0) ? H_new : (H_new / pad_multiple + 1) * pad_multiple;
        int W_pad = (W_new % pad_multiple == 0) ? W_new : (W_new / pad_multiple + 1) * pad_multiple;
        int pad_top = (H_pad - H_new) / 2;
        int pad_left = (W_pad - W_new) / 2;

        cv::Mat padded = cv::Mat::zeros(H_pad, W_pad, CV_8UC3);
        resized.copyTo(padded(cv::Rect(pad_left, pad_top, W_new, H_new)));

        res.img = padded;
        res.scale = scale;
        res.pad_left = pad_left;
        res.pad_top = pad_top;
        return res;
    }

    // ─── HccePose preprocessing ────────────────────────────────────────────────────
    std::vector<float> Preprocessing::prepareForHccePose(const cv::Mat &crop_rgb) {
        // Očekáváme 256×256 uint8 RGB
        CV_Assert(crop_rgb.type() == CV_8UC3);

        const int H = crop_rgb.rows, W = crop_rgb.cols;
        std::vector<float> chw(3 * H * W);

        // Rozsplit na kanály, normalizuj, ulož v CHW pořadí
        // Python: img.flip(dims=[1]) = BGR→RGB swap (ale my už máme RGB)
        // Normalizace: (pixel/255 - mean) / std
        for (int r = 0; r < H; r++) {
            const uchar *row = crop_rgb.ptr<uchar>(r);
            for (int c = 0; c < W; c++) {
                for (int ch = 0; ch < 3; ch++) {
                    float val = row[c * 3 + ch] / 255.0f;
                    val = (val - MEAN[ch]) / STD[ch];
                    chw[ch * H * W + r * W + c] = val;
                }
            }
        }
        return chw;
    }

    // ─── Pomocná ──────────────────────────────────────────────────────────────────
    cv::Mat Preprocessing::bgrToRgbFloat(const cv::Mat &img_bgr) {
        cv::Mat rgb;
        cv::cvtColor(img_bgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat flt;
        rgb.convertTo(flt, CV_32F, 1.0 / 255.0);
        return flt;
    }

    std::vector<float> Preprocessing::floatMatToCHW(const cv::Mat& img_float) {
        const int H = img_float.rows, W = img_float.cols;
        std::vector<float> chw(3 * H * W);
        for (int r = 0; r < H; r++) {
            const float* row = img_float.ptr<float>(r);
            for (int c = 0; c < W; c++) {
                chw[0 * H * W + r * W + c] = row[c * 3 + 0];
                chw[1 * H * W + r * W + c] = row[c * 3 + 1];
                chw[2 * H * W + r * W + c] = row[c * 3 + 2];
            }
        }
        return chw;
    }
} // namespace hcce
