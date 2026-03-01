#include "ultradensesampler.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>

namespace hcce {

// ─── k-d tree (zjednodušená verze pro výpočet průměrné NN vzdálenosti) ───────
// Pro přesný výpočet je možné použít nanoflann (header-only)
// Zde používáme náhodný podvzorkování pro rychlost
float UltraDenseSampler::computeAverageNNDistance(
    const std::vector<cv::Point3f>& pts, int max_samples)
{
    if (pts.size() < 2) return 1.0f; // fallback

    // Podvzorkování pokud je příliš mnoho bodů
    std::vector<int> indices(pts.size());
    std::iota(indices.begin(), indices.end(), 0);

    if ((int)pts.size() > max_samples) {
        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(max_samples);
    }

    float total_dist = 0.0f;
    int   count      = 0;

    for (int idx : indices) {
        float best = std::numeric_limits<float>::max();
        for (int jdx : indices) {
            if (idx == jdx) continue;
            float dx = pts[idx].x - pts[jdx].x;
            float dy = pts[idx].y - pts[jdx].y;
            float dz = pts[idx].z - pts[jdx].z;
            float d  = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d < best) best = d;
        }
        if (best < std::numeric_limits<float>::max()) {
            total_dist += best;
            count++;
        }
    }

    return (count > 0) ? total_dist / count : 1.0f;
}

// ─── Ultra-dense sampling (rovnice 1, 2, 3) ───────────────────────────────────
DenseCorrespondences UltraDenseSampler::sample(
    const std::vector<cv::Point2f>& pixels_2d,
    const std::vector<cv::Point3f>& front_3d,
    const std::vector<cv::Point3f>& back_3d)
{
    CV_Assert(pixels_2d.size() == front_3d.size());
    CV_Assert(pixels_2d.size() == back_3d.size());

    DenseCorrespondences result;

    // ── 1. Spočítej d (průměrná NN vzdálenost v 3D) ───────────────────────────
    // Kombinuj front a back body pro reprezentativní výpočet
    std::vector<cv::Point3f> all_pts;
    all_pts.insert(all_pts.end(), front_3d.begin(), front_3d.end());
    all_pts.insert(all_pts.end(), back_3d.begin(),  back_3d.end());

    float d_avg = computeAverageNNDistance(all_pts, 500);
    if (d_avg < 1e-6f) d_avg = 1.0f; // ochrana před dělením nulou

    // ── 2. Pro každý pixel přidej front, back + interpolované body ────────────
    for (size_t i = 0; i < pixels_2d.size(); i++) {
        const cv::Point2f& p2d = pixels_2d[i];
        const cv::Point3f& q1  = front_3d[i];  // přední plocha
        const cv::Point3f& q2  = back_3d[i];   // zadní plocha

        // Přidej front bod (Pf, Qf)
        result.pts2d.push_back(p2d);
        result.pts3d.push_back(q1);

        // Přidej back bod (Pb, Qb)
        result.pts2d.push_back(p2d);
        result.pts3d.push_back(q2);

        // ── Rovnice 1: n = floor(||q1-q2||_2 / d) ─────────────────────────
        float dx = q1.x - q2.x;
        float dy = q1.y - q2.y;
        float dz = q1.z - q2.z;
        float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        int   n    = (int)(dist / d_avg);

        // ── Rovnice 2: s(q1, q2, a) = a*q1 + (1-a)*q2 ───────────────────
        // a ∈ {t/(n+1) | t = 1, 2, ..., n}
        for (int t = 1; t <= n; t++) {
            float a = (float)t / (float)(n + 1);
            cv::Point3f q_mid(
                a * q1.x + (1.0f - a) * q2.x,
                a * q1.y + (1.0f - a) * q2.y,
                a * q1.z + (1.0f - a) * q2.z
            );
            // Pf = Pb = Pm → sdílí stejné 2D souřadnice
            result.pts2d.push_back(p2d);
            result.pts3d.push_back(q_mid);
        }
    }

    return result;
}

} // namespace hcce