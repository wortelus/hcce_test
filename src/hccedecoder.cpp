#include "hccedecoder.h"
#include <cmath>
#include <cassert>

namespace hcce {

HcceDecoder::HcceDecoder(const ObjBounds& bounds)
    : bounds_(bounds)
{}

// ─── Dekódování jedné složky (rovnice 6 + 4) ────────────────────────────────
// continuous_codes: pole n_levels floatů (výstup sítě pro jednu složku)
// Vrací hodnotu složky v rozsahu [0, 1]
float HcceDecoder::decodeComponent(const float* continuous_codes, int n_levels)
{
    // ── Krok 1: Continuous → Binary (rovnice 6) ──────────────────────────────
    // B_{i,k} = g(C_{i,k})           pokud B_{i-1,k} = 0
    //         = 1 - g(C_{i,k})       pokud B_{i-1,k} = 1   (mirror kompenzace)
    std::vector<int> B(n_levels);

    // Level 1: přímá binarizace (rovnice před eq.6)
    B[0] = binarize(continuous_codes[0]);

    for (int i = 1; i < n_levels; i++) {
        int g_val = binarize(continuous_codes[i]);
        if (B[i - 1] == 0)
            B[i] = g_val;
        else
            B[i] = 1 - g_val;  // zrušení mirror operace z encodingu
    }

    // ── Krok 2: Binary → souřadnice (rovnice 4) ──────────────────────────────
    // x_k ≈ Σ_{i=1}^{8} 2^{-i} * B_{i,k}
    float value = 0.0f;
    for (int i = 0; i < n_levels; i++) {
        value += std::pow(2.0f, -(float)(i + 1)) * (float)B[i];
    }

    // Clamp na [0, 1] pro numerickou stabilitu
    return std::max(0.0f, std::min(1.0f, value));
}

// ─── Dekódování jednoho pixelu ────────────────────────────────────────────────
// codes: 24 floatů = [x0..x7, y0..y7, z0..z7]
// (pořadí: 8 úrovní pro x, pak 8 pro y, pak 8 pro z)
cv::Point3f HcceDecoder::decodePixel(const float* codes) const
{
    float nx = decodeComponent(codes + 0,  8);  // kanály 0..7  = x složka
    float ny = decodeComponent(codes + 8,  8);  // kanály 8..15 = y složka
    float nz = decodeComponent(codes + 16, 8);  // kanály 16..23= z složka
    return cv::Point3f(nx, ny, nz);
}

// ─── De-normalizace [0,1] → reálné mm ────────────────────────────────────────
cv::Point3f HcceDecoder::denormalize(const cv::Point3f& n) const
{
    return cv::Point3f(
        n.x * bounds_.size_x + bounds_.min_x,
        n.y * bounds_.size_y + bounds_.min_y,
        n.z * bounds_.size_z + bounds_.min_z
    );
}

// ─── Hlavní dekódovací funkce ─────────────────────────────────────────────────
DecodedCoords HcceDecoder::decode(const NetworkOutput& net_out, float mask_threshold)
{
    const int H = net_out.mask.rows;  // 128
    const int W = net_out.mask.cols;  // 128

    DecodedCoords result;

    for (int r = 0; r < H; r++) {
        const float* mask_row = net_out.mask.ptr<float>(r);

        for (int c = 0; c < W; c++) {
            // Přeskočit pixely mimo masku
            if (mask_row[c] <= mask_threshold) continue;

            // ── Front kódy pro tento pixel ────────────────────────────────
            const float* front = net_out.front_codes.ptr<float>(r) + c * 24;
            cv::Point3f  f_norm = decodePixel(front);
            cv::Point3f  f_3d   = denormalize(f_norm);

            // ── Back kódy pro tento pixel ──────────────────────────────────
            const float* back  = net_out.back_codes.ptr<float>(r) + c * 24;
            cv::Point3f  b_norm = decodePixel(back);
            cv::Point3f  b_3d   = denormalize(b_norm);

            // ── 2D souřadnice v crop prostoru (128×128) ───────────────────
            cv::Point2f pixel((float)c, (float)r);

            result.pixels_2d.push_back(pixel);
            result.front_3d.push_back(f_3d);
            result.back_3d.push_back(b_3d);
        }
    }

    return result;
}

} // namespace hcce