#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace hcce {

    class Visualizer {
    public:
        // Nakreslí 2D detekce (bbox + label) na obraz
        static void drawDetections(cv::Mat& img,
                                    const std::vector<Detection>& dets);

        // Vykreslí 3D bounding box objektu přes image (drátěný model)
        static void draw3DBBox(cv::Mat& img,
                                const cv::Mat& R,
                                const cv::Mat& t,
                                const cv::Mat& K,
                                const ObjectInfo& obj_info,
                                const cv::Scalar& color = cv::Scalar(0, 255, 0));

        // Vykreslí souřadnicové osy objektu
        static void drawAxes(cv::Mat& img,
                              const cv::Mat& R,
                              const cv::Mat& t,
                              const cv::Mat& K,
                              float axis_length = 50.0f);

        // Vytvoří vizualizaci predikovaných front/back 3D souřadnic (RGB)
        // coords: normalizované [0,1]^3 souřadnice
        static cv::Mat visualizeCoords(const cv::Mat& coords_hwc3,
                                        const cv::Mat& mask);
    };

} // namespace hcce