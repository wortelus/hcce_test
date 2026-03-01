#include "visualizer.h"
#include <random>

namespace hcce {

void Visualizer::drawDetections(cv::Mat& img, const std::vector<Detection>& dets)
{
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(50, 255);

    for (const auto& d : dets) {
        cv::Scalar color(dist(rng), dist(rng), dist(rng));
        cv::rectangle(img, d.bbox, color, 2);
        std::string label = "obj_" + std::to_string(d.obj_id)
                          + " " + std::to_string((int)(d.confidence * 100)) + "%";
        cv::putText(img, label,
                    cv::Point(d.bbox.x, d.bbox.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

void Visualizer::draw3DBBox(cv::Mat& img,
                              const cv::Mat& R, const cv::Mat& t,
                              const cv::Mat& K,
                              const ObjectInfo& obj_info,
                              const cv::Scalar& color)
{
    // 8 rohů bounding boxu
    const auto& corners = obj_info.bbox3d_corners;
    std::vector<cv::Point2f> projected;
    cv::projectPoints(corners, R, t, K, cv::Mat(), projected);

    // 12 hran krychle
    static const int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0},  // spodní plocha
        {4,5},{5,6},{6,7},{7,4},  // horní plocha
        {0,4},{1,5},{2,6},{3,7}   // svislé hrany
    };
    for (auto& e : edges) {
        cv::line(img,
                 cv::Point((int)projected[e[0]].x, (int)projected[e[0]].y),
                 cv::Point((int)projected[e[1]].x, (int)projected[e[1]].y),
                 color, 2);
    }
}

void Visualizer::drawAxes(cv::Mat& img,
                            const cv::Mat& R, const cv::Mat& t,
                            const cv::Mat& K, float axis_length)
{
    std::vector<cv::Point3f> axis_pts = {
        {0,0,0}, {axis_length,0,0}, {0,axis_length,0}, {0,0,axis_length}
    };
    std::vector<cv::Point2f> proj;
    cv::projectPoints(axis_pts, R, t, K, cv::Mat(), proj);

    cv::line(img, proj[0], proj[1], cv::Scalar(0,0,255), 3);   // X = červená
    cv::line(img, proj[0], proj[2], cv::Scalar(0,255,0), 3);   // Y = zelená
    cv::line(img, proj[0], proj[3], cv::Scalar(255,0,0), 3);   // Z = modrá
}

cv::Mat Visualizer::visualizeCoords(const cv::Mat& coords_hwc3, const cv::Mat& mask)
{
    CV_Assert(coords_hwc3.channels() == 3);
    cv::Mat vis;
    coords_hwc3.convertTo(vis, CV_8UC3, 255.0);
    // Zamaskuj background
    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask.at<float>(r, c) <= 0.0f)
                vis.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
        }
    }
    return vis;
}

} // namespace hcce