#pragma once
#include "types.h"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace hcce {

    class YoloDetector {
    public:
        YoloDetector(const std::string& model_path,
                     const std::vector<int>& obj_id_list,
                     const std::string& cuda_device = "0");

        // Detekuje objekty v obrazu
        // img_bgr: vstupní BGR obraz (libovolná velikost)
        // Vrací detekce ve souřadnicích PŮVODNÍHO obrazu
        std::vector<Detection> detect(const cv::Mat& img_bgr,
                                       float conf_threshold = 0.85f,
                                       float iou_threshold  = 0.50f,
                                       int   max_det        = 200);

    private:
        Ort::Env               env_;
        Ort::Session           session_{nullptr};
        Ort::SessionOptions    session_opts_;
        std::vector<int>       obj_id_list_;

        // NMS (non-maximum suppression)
        static std::vector<Detection> nms(std::vector<Detection>& dets,
                                           float iou_threshold);
        static float iou(const cv::Rect& a, const cv::Rect& b);

        // Parsování raw YOLO output tensoru
        // YOLO11 výstup: [1, 4+num_classes, num_anchors]
        std::vector<Detection> parseYoloOutput(
            const float* data, int num_anchors, int num_classes,
            int img_w, int img_h,
            float conf_threshold,
            float scale, int pad_left, int pad_top);
    };

} // namespace hcce