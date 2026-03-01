#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>

namespace hcce {

    // Implementuje ultra-dense sampling z paperu (Section 3.1)
    // Rovnice 1: n = floor(||q1-q2||_2 / d)
    // Rovnice 2: s(q1, q2, a) = a*q1 + (1-a)*q2
    // Rovnice 3: Pu = {Pb, Pf, Pm}, Qu = {Qb, Qf, Qm}
    class UltraDenseSampler {
    public:
        // Vstup: dekódované front a back 3D souřadnice
        // Výstup: ultra-dense 2D-3D korespondence (front + back + interpolované)
        // pixels_2d musí odpovídat front_3d i back_3d (indexově)
        static DenseCorrespondences sample(
            const std::vector<cv::Point2f>& pixels_2d,
            const std::vector<cv::Point3f>& front_3d,
            const std::vector<cv::Point3f>& back_3d
        );

    private:
        // Spočítá průměrnou vzdálenost nejbližšího souseda metodou k-d stromu
        // (nebo zjednodušeně z náhodného podvzorku pro rychlost)
        static float computeAverageNNDistance(
            const std::vector<cv::Point3f>& pts,
            int max_samples = 1000
        );
    };

} // namespace hcce