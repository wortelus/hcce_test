#pragma once
#include <opencv2/opencv.hpp>

namespace hcce {

    // Replikuje Python funkci crop_trans_batch_hccepose()
    // Vypočítá homografii pro crop bbox → čtvercový výřez
    class CropTransform {
    public:
        struct CropInfo {
            cv::Mat  M;         // 3×3 homografie (float64): orig → crop
            cv::Mat  M_inv;     // 3×3 inverzní (crop → orig)
            cv::Rect src_bbox;  // bbox v orig obrazu (před padding)
        };

        // Spočítá crop transformaci pro jeden bbox
        // bbox: [x, y, w, h] v pixelech
        // out_size: výstupní rozlišení (default 256×256)
        // padding_ratio: 1.5 (jak v Python kódu)
        static CropInfo compute(const cv::Rect& bbox,
                                 int out_size     = 256,
                                 float pad_ratio  = 1.5f);

        // Aplikuje crop transformaci na obraz
        // Vstup: orig RGB/BGR obraz, CropInfo
        // Výstup: crop out_size×out_size
        static cv::Mat warp(const cv::Mat& img, const CropInfo& info, int out_size = 256);

        // Spočítá transformaci v polovičním rozlišení (pro 128×128 výstup sítě)
        // Potřebné pro zpětnou transformaci predikovaných masek a kódů
        static CropInfo computeHalf(const cv::Rect& bbox,
                                      float pad_ratio = 1.5f);

        // Transformuje 2D bod z crop prostoru zpět do orig obrazu
        static cv::Point2f transformPoint(const cv::Point2f& pt, const cv::Mat& M_inv);

        // Transformuje celý výstup sítě (128×128) zpět do orig rozlišení
        // pomocí inverzní homografie
        static cv::Mat warpBack(const cv::Mat& crop_output,
                                 const CropInfo& info_128,
                                 int orig_h, int orig_w,
                                 int interp = cv::INTER_NEAREST);
    };

} // namespace hcce