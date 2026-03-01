#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace hcce {

// ─── Výsledek YOLO detekce ────────────────────────────────────────────────────
struct Detection {
    cv::Rect  bbox;       // x,y,w,h v pixelech
    float     confidence;
    int       class_id;   // index v obj_id_list
    int       obj_id;     // skutečné ID objektu
};

// ─── Informace o objektu z models_info.json ───────────────────────────────────
struct ObjectInfo {
    int   obj_id;
    float min_x, min_y, min_z;
    float size_x, size_y, size_z;
    // 3D bounding box rohy (8 bodů)
    std::vector<cv::Point3f> bbox3d_corners;
};

// ─── Výstup HccePose sítě (surová data) ──────────────────────────────────────
// Tvar tensoru: [49, 128, 128]
// kanály  0..23 = front HCCE (8 úrovní × 3 složky x,y,z)
// kanály 24..47 = back  HCCE (8 úrovní × 3 složky x,y,z)
// kanál   48    = maska
struct NetworkOutput {
    cv::Mat mask;             // [128,128] float32
    cv::Mat front_codes;      // [128,128,24] float32
    cv::Mat back_codes;       // [128,128,24] float32
};

// ─── Dekódované 3D souřadnice ─────────────────────────────────────────────────
struct DecodedCoords {
    std::vector<cv::Point2f> pixels_2d;   // souřadnice v CROPPED obrazu
    std::vector<cv::Point3f> front_3d;
    std::vector<cv::Point3f> back_3d;
};

// ─── Ultra-dense 2D-3D korespondence ─────────────────────────────────────────
struct DenseCorrespondences {
    std::vector<cv::Point2f> pts2d;
    std::vector<cv::Point3f> pts3d;
};

// ─── Výsledná 6D póza ─────────────────────────────────────────────────────────
struct PoseResult {
    int       obj_id;
    cv::Mat   R;           // 3×3 rotační matice (float64)
    cv::Mat   t;           // 3×1 translace (float64)
    float     confidence;
    bool      valid;
};

// ─── Informace o kameře ───────────────────────────────────────────────────────
struct CameraIntrinsics {
    double fx, fy, cx, cy;

    cv::Mat toMat() const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0,0) = fx; K.at<double>(0,2) = cx;
        K.at<double>(1,1) = fy; K.at<double>(1,2) = cy;
        return K;
    }
};

} // namespace hcce