#include "hcceposeestimator.h"
#include <stdexcept>
#include <cstring>

namespace hcce {
    HccePoseEstimator::HccePoseEstimator(const std::string &model_path,
                                         const std::string &cuda_device)
        : env_(ORT_LOGGING_LEVEL_WARNING, "HccePose") {
        session_opts_.SetIntraOpNumThreads(4);
        session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = std::stoi(cuda_device);
        session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
#endif

        // DEBUG - ukaž přesně co dostáváme
        std::cerr << "HccePoseEstimator loading: [" << model_path << "]\n";

        std::wstring model_path_w(model_path.begin(), model_path.end());
        session_ = Ort::Session(env_, model_path_w.c_str(), session_opts_);
    }

    NetworkOutput HccePoseEstimator::infer(const std::vector<float> &crop_chw) {
        CV_Assert((int)crop_chw.size() == 3 * INPUT_H * INPUT_W);

        // ── 1. Vstupní tensor ────────────────────────────────────────────────────
        std::array<int64_t, 4> input_shape = {1, 3, INPUT_H, INPUT_W};
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Potřebujeme non-const pointer pro ONNX API
        std::vector<float> input_copy = crop_chw;
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, input_copy.data(), input_copy.size(),
            input_shape.data(), 4);

        // ── 2. Inference ─────────────────────────────────────────────────────────
        const char *input_names[] = {"input_rgb"};
        const char *output_names[] = {"pred_mask", "pred_codes"};

        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    input_names, &input_tensor, 1,
                                    output_names, 2);

        // ── 3. Parsování výstupu [1, 49, 128, 128] ───────────────────────────────
        // const float* data = outputs[0].GetTensorData<float>();
        // outputs[0] = pred_mask  [1, 1, 128, 128]
        const float *mask_data = outputs[0].GetTensorData<float>();
        // outputs[1] = pred_codes [1, 48, 128, 128]
        const float *codes_data = outputs[1].GetTensorData<float>();

        return parseOutput(mask_data, codes_data);
    }

    NetworkOutput HccePoseEstimator::parseOutput(const float *mask_data, const float *codes_data) {
        const int SPATIAL = OUTPUT_H * OUTPUT_W;
        NetworkOutput out;

        // Maska z prvního výstupu
        out.mask = cv::Mat(OUTPUT_H, OUTPUT_W, CV_32F);
        std::memcpy(out.mask.data, mask_data, SPATIAL * sizeof(float));

        // Front kódy: kanály 0..23 z codes_data
        out.front_codes = cv::Mat(OUTPUT_H, OUTPUT_W, CV_32FC(24));
        for (int c = 0; c < 24; c++) {
            const float *src = codes_data + c * SPATIAL;
            for (int r = 0; r < OUTPUT_H; r++)
                for (int col = 0; col < OUTPUT_W; col++)
                    out.front_codes.at<cv::Vec<float, 24> >(r, col)[c] = src[r * OUTPUT_W + col];
        }

        // Back kódy: kanály 24..47 z codes_data
        out.back_codes = cv::Mat(OUTPUT_H, OUTPUT_W, CV_32FC(24));
        for (int c = 0; c < 24; c++) {
            const float *src = codes_data + (24 + c) * SPATIAL;
            for (int r = 0; r < OUTPUT_H; r++)
                for (int col = 0; col < OUTPUT_W; col++)
                    out.back_codes.at<cv::Vec<float, 24> >(r, col)[c] = src[r * OUTPUT_W + col];
        }

        return out;
    }
} // namespace hcce
