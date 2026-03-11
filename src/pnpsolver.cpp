#include "pnpsolver.h"
#include "croptransform.h"

namespace hcce {
    PnPSolver::PnPSolver(const Config &cfg) : cfg_(cfg) {
    }

    PoseResult  PnPSolver::solve(
        const std::vector<cv::Point3f> &front_3d,
        const std::vector<cv::Point3f> &back_3d,
        const std::vector<cv::Point2f> &pts2d_128,
        const cv::Mat &K,
        const CropTransform::CropInfo &crop_info_128) const {
        PoseResult result;
        result.valid = false;
        result.R = cv::Mat::eye(3, 3, CV_64F);
        result.t = cv::Mat::zeros(3, 1, CV_64F);

        const int N = (int) front_3d.size();
        if (N < cfg_.min_inliers) return result;

        // Transformace 2D bodů do orig prostoru
        std::vector<cv::Point2f> pts2d_orig(N);
        for (int i = 0; i < N; i++)
            pts2d_orig[i] = CropTransform::transformPoint(pts2d_128[i], crop_info_128.M_inv);

        // bf: alternuje front/back
        std::vector<cv::Point3f> bf_3d(N);
        for (int i = 0; i < N; i++)
            bf_3d[i] = (i % 2 == 0) ? back_3d[i] : front_3d[i];

        // Spusť 3 varianty: front-only, back-only, bf
        struct Candidate {
            std::vector<cv::Point3f> pts3d;
        };
        std::vector<Candidate> candidates = {
            {front_3d},
            {back_3d},
            {bf_3d},
        };

        PoseResult best;
        best.valid = false;
        best.inliers = 0;
        best.R = cv::Mat::eye(3, 3, CV_64F);
        best.t = cv::Mat::zeros(3, 1, CV_64F);

        for (auto &cand: candidates) {
            cv::Mat rvec, tvec;
            std::vector<int> inliers_cv;
            bool ok = cv::solvePnPRansac(
                cand.pts3d, pts2d_orig, K, cv::Mat(),
                rvec, tvec, false,
                cfg_.ransac_iterations, cfg_.reproj_threshold, 0.99,
                inliers_cv, cv::SOLVEPNP_EPNP);

            if (!ok || (int) inliers_cv.size() < cfg_.min_inliers) continue;

            // Refinement
            std::vector<cv::Point3f> inlier_3d;
            std::vector<cv::Point2f> inlier_2d;
            for (int idx: inliers_cv) {
                inlier_3d.push_back(cand.pts3d[idx]);
                inlier_2d.push_back(pts2d_orig[idx]);
            }
            cv::solvePnP(inlier_3d, inlier_2d, K, cv::Mat(),
                         rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

            if ((int) inliers_cv.size() > best.inliers) {
                cv::Rodrigues(rvec, best.R);
                best.t = tvec;
                best.valid = true;
                best.inliers = (int) inliers_cv.size();
            }
        }

        return best;
    }
} // namespace hcce
