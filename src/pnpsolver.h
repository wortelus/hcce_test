#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>

#include "croptransform.h"

namespace hcce {

    // Custom RANSAC-PnP implementující omezení z paperu (Section 3.1):
    // "V každé iteraci RANSAC smí být vybrán maximálně 1 bod na pixel"
    // Počet iterací: 150, reprojekční práh: 2px
    class PnPSolver {
    public:
        struct Config {
            int   ransac_iterations = 150;
            float reproj_threshold  = 2.0f;   // pixely
            int   min_inliers       = 4;
        };

        PnPSolver(const Config& cfg = Config{});

        // Hlavní funkce: vypočítá 6D pózu z ultra-dense 2D-3D korespondencí
        // Vstup:
        //   corr       - ultra-dense 2D-3D korespondence (Pu, Qu)
        //   K          - matice kamery (3×3 float64)
        //   crop_info  - informace o cropu (pro transformaci 2D souřadnic zpět do orig)
        // Výstup: PoseResult s R, t
        PoseResult solve(const DenseCorrespondences& corr,
                         const cv::Mat& K,
                         const CropTransform::CropInfo& crop_info_128);

    private:
        Config cfg_;

        // Vyber 4 náhodné body tak, aby každý měl unikátní 2D souřadnici
        // (implementace per-pixel omezení z paperu)
        std::vector<int> sampleUniquePixels(
            const std::vector<cv::Point2f>& pts2d,
            int n = 4) const;

        // Vypočítá reprojekční chybu
        float reprojectionError(
            const cv::Point3f& pt3d,
            const cv::Point2f& pt2d,
            const cv::Mat& R, const cv::Mat& t,
            const cv::Mat& K) const;

        // Spočítá počet inlierů pro danou pózu
        std::vector<int> computeInliers(
            const std::vector<cv::Point3f>& pts3d,
            const std::vector<cv::Point2f>& pts2d,
            const cv::Mat& R, const cv::Mat& t,
            const cv::Mat& K,
            float threshold) const;
    };

    // Nutné includovat CropTransform
#include "CropTransform.h"

} // namespace hcce