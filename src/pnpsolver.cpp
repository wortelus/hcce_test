#include "pnpsolver.h"
#include "croptransform.h"
#include <random>
#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace hcce {

PnPSolver::PnPSolver(const Config& cfg) : cfg_(cfg) {}

// ─── Per-pixel RANSAC vzorkování ─────────────────────────────────────────────
// Klíčová část: z bodů se stejnou 2D pozicí vyber max 1
std::vector<int> PnPSolver::sampleUniquePixels(
    const std::vector<cv::Point2f>& pts2d, int n) const
{
    static std::mt19937 rng(std::random_device{}());

    // Vytvoř mapu pixel → seznam indexů bodů
    // Klíč: (int_x, int_y) → unikátní pixel
    std::unordered_map<uint64_t, std::vector<int>> pixel_map;
    for (int i = 0; i < (int)pts2d.size(); i++) {
        int px = (int)std::round(pts2d[i].x);
        int py = (int)std::round(pts2d[i].y);
        uint64_t key = ((uint64_t)(px & 0xFFFFFFFF) << 32) | (uint64_t)(py & 0xFFFFFFFF);
        pixel_map[key].push_back(i);
    }

    // Sesbírej unikátní pixely
    std::vector<uint64_t> unique_pixels;
    unique_pixels.reserve(pixel_map.size());
    for (auto& [k, _] : pixel_map)
        unique_pixels.push_back(k);

    if ((int)unique_pixels.size() < n)
        return {}; // nedostatek unikátních pixelů

    // Náhodně vyber n unikátních pixelů
    std::shuffle(unique_pixels.begin(), unique_pixels.end(), rng);
    unique_pixels.resize(n);

    // Pro každý vybraný pixel, náhodně vyber jeden z jeho 3D bodů
    std::vector<int> selected;
    for (auto key : unique_pixels) {
        auto& candidates = pixel_map[key];
        std::uniform_int_distribution<int> dist(0, (int)candidates.size() - 1);
        selected.push_back(candidates[dist(rng)]);
    }
    return selected;
}

// ─── Reprojekční chyba pro jeden bod ─────────────────────────────────────────
float PnPSolver::reprojectionError(
    const cv::Point3f& pt3d, const cv::Point2f& pt2d,
    const cv::Mat& R, const cv::Mat& t, const cv::Mat& K) const
{
    std::vector<cv::Point3f> obj = {pt3d};
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, R, t, K, cv::Mat(), proj);
    float dx = proj[0].x - pt2d.x;
    float dy = proj[0].y - pt2d.y;
    return std::sqrt(dx*dx + dy*dy);
}

// ─── Inlieři ──────────────────────────────────────────────────────────────────
std::vector<int> PnPSolver::computeInliers(
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    const cv::Mat& R, const cv::Mat& t,
    const cv::Mat& K, float threshold) const
{
    std::vector<cv::Point2f> projected;
    cv::projectPoints(pts3d, R, t, K, cv::Mat(), projected);

    std::vector<int> inliers;
    for (int i = 0; i < (int)pts3d.size(); i++) {
        float dx = projected[i].x - pts2d[i].x;
        float dy = projected[i].y - pts2d[i].y;
        if (std::sqrt(dx*dx + dy*dy) < threshold)
            inliers.push_back(i);
    }
    return inliers;
}

// ─── Hlavní RANSAC-PnP ────────────────────────────────────────────────────────
PoseResult PnPSolver::solve(
    const DenseCorrespondences& corr,
    const cv::Mat& K,
    const CropTransform::CropInfo& crop_info_128)
{
    PoseResult result;
    result.valid  = false;
    result.R      = cv::Mat::eye(3, 3, CV_64F);
    result.t      = cv::Mat::zeros(3, 1, CV_64F);

    const int N = (int)corr.pts3d.size();
    if (N < cfg_.min_inliers) return result;

    // ── Transformace 2D bodů z crop(128) prostoru do orig obrazu ──────────────
    std::vector<cv::Point2f> pts2d_orig(N);
    for (int i = 0; i < N; i++)
        pts2d_orig[i] = CropTransform::transformPoint(corr.pts2d[i], crop_info_128.M_inv);

    const std::vector<cv::Point3f>& pts3d = corr.pts3d;

    // ── Custom RANSAC se per-pixel omezením ───────────────────────────────────
    cv::Mat best_R, best_t;
    int     best_inlier_count = 0;

    for (int iter = 0; iter < cfg_.ransac_iterations; iter++) {
        // Vyber 4 body s unikátními 2D pixely
        auto indices = sampleUniquePixels(pts2d_orig, 4);
        if (indices.empty()) continue;

        std::vector<cv::Point3f> sample_3d;
        std::vector<cv::Point2f> sample_2d;
        for (int idx : indices) {
            sample_3d.push_back(pts3d[idx]);
            sample_2d.push_back(pts2d_orig[idx]);
        }

        // EPnP pro minimální sadu
        cv::Mat rvec, tvec;
        try {
            bool ok = cv::solvePnP(sample_3d, sample_2d, K, cv::Mat(),
                                    rvec, tvec, false, cv::SOLVEPNP_EPNP);
            if (!ok) continue;
        } catch (...) { continue; }

        // Spočítej inlieři
        cv::Mat R_iter;
        cv::Rodrigues(rvec, R_iter);
        auto inliers = computeInliers(pts3d, pts2d_orig, R_iter, tvec, K,
                                       cfg_.reproj_threshold);

        if ((int)inliers.size() > best_inlier_count) {
            best_inlier_count = (int)inliers.size();
            best_R = R_iter.clone();
            best_t = tvec.clone();
        }
    }

    if (best_inlier_count < cfg_.min_inliers) return result;

    // ── Refinement VVS (iterativní) na všech inlierech ─────────────────────
    // Znovu spočítej inlieře z nejlepší pózy
    auto final_inliers = computeInliers(pts3d, pts2d_orig, best_R, best_t, K,
                                         cfg_.reproj_threshold);

    if ((int)final_inliers.size() >= cfg_.min_inliers) {
        std::vector<cv::Point3f> inlier_3d;
        std::vector<cv::Point2f> inlier_2d;
        for (int idx : final_inliers) {
            inlier_3d.push_back(pts3d[idx]);
            inlier_2d.push_back(pts2d_orig[idx]);
        }

        // Iterativní refinement (VVS = Virtual Visual Servoing)
        cv::Mat rvec_ref;
        cv::Rodrigues(best_R, rvec_ref);
        cv::solvePnP(inlier_3d, inlier_2d, K, cv::Mat(),
                     rvec_ref, best_t, true,   // useExtrinsicGuess=true
                     cv::SOLVEPNP_ITERATIVE);
        cv::Rodrigues(rvec_ref, best_R);
    }

    result.R     = best_R;
    result.t     = best_t;
    result.valid = true;
    return result;
}

} // namespace hcce