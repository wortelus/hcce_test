#pragma once
#include "Types.h"
#include <opencv2/opencv.hpp>

namespace hcce {

    // Implementuje HCCE dekódování podle paperu (Section 3.2.2)
    // Rovnice 4: xk ≈ Σ 2^(-i) * Bx_{i,k}
    // Rovnice 5: continuous → hierarchical continuous (jen pro encoding, ne zde)
    // Rovnice 6: continuous → binary s mirror kompenzací
    class HcceDecoder {
    public:
        // Informace o objektu pro de-normalizaci souřadnic
        struct ObjBounds {
            float min_x, min_y, min_z;
            float size_x, size_y, size_z;
        };

        HcceDecoder(const ObjBounds& bounds);

        // Hlavní funkce: dekóduje NetworkOutput → DecodedCoords
        // mask_threshold: práh pro klasifikaci pixelu jako objekt (default 0.0)
        DecodedCoords decode(const NetworkOutput& net_out,
                              float mask_threshold = 0.0f);

        // Dekóduje jediný pixel
        // codes: 24 floatů (8 úrovní × 3 složky: xxxxx...yyyyy...zzzzz...)
        // Vrací normalizovanou souřadnici [0,1]^3
        cv::Point3f decodePixel(const float* codes) const;

        // De-normalizuje souřadnici [0,1] → reálné mm
        cv::Point3f denormalize(const cv::Point3f& normalized) const;

        // Dekóduje jednu složku (8 continuous kódů → float hodnota [0,1])
        // Implementuje rovnice 6 + 4
        static float decodeComponent(const float* continuous_codes, int n_levels = 8);
    private:
        ObjBounds bounds_;

        // Binarizační funkce g(t): 0 pokud t<0.5, jinak 1
        static inline int binarize(float t) { return t >= 0.0f ? 1 : 0; }
    };

} // namespace hcce