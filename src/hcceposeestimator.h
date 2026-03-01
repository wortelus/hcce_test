#pragma once
#include "types.h"
#include <onnxruntime_cxx_api.h>
#include <string>

namespace hcce {

    // Wrapper kolem HccePose ONNX modelu
    // Vstup:  [1, 3, 256, 256] float32 (ImageNet normalizované)
    // Výstup: [1, 49, 128, 128] float32
    //   kanály  0..23 = front HCCE kódy
    //   kanály 24..47 = back  HCCE kódy
    //   kanál   48    = maska
    class HccePoseEstimator {
    public:
        HccePoseEstimator(const std::string& model_path,
                          const std::string& cuda_device = "0");

        // Spustí inferenci
        // crop_chw: float32 tensor [3 * 256 * 256] v CHW pořadí
        // Vrací NetworkOutput (maska + front/back kódy v rozlišení 128×128)
        NetworkOutput infer(const std::vector<float>& crop_chw);

    private:
        Ort::Env            env_;
        Ort::Session        session_{nullptr};
        Ort::SessionOptions session_opts_;

        static constexpr int INPUT_H  = 256;
        static constexpr int INPUT_W  = 256;
        static constexpr int OUTPUT_H = 128;
        static constexpr int OUTPUT_W = 128;
        static constexpr int NUM_CHANNELS = 49; // 24 + 24 + 1

        // Parsování výstupního tensoru do NetworkOutput
        static NetworkOutput parseOutput(const float *mask_data, const float *codes_data);
    };

} // namespace hcce