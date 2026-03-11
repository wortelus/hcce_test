#pragma once
#include "types.h"
#include "croptransform.h"
#include <opencv2/opencv.hpp>

namespace hcce {
    struct Config {
        int   ransac_iterations = 150;
        float reproj_threshold  = 1.5f;
        int   min_inliers       = 4;
    };

    class PnPSolver {
    public:
        PnPSolver(const Config& cfg = Config());

        PoseResult solve(const std::vector<cv::Point3f>& front_3d,
                         const std::vector<cv::Point3f>& back_3d,
                         const std::vector<cv::Point2f>& pts2d_128,
                         const cv::Mat& K,
                         const CropTransform::CropInfo& crop_info_128) const;

    private:
        Config cfg_;
    };

} // namespace hcce