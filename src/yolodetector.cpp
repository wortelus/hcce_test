#include "yolodetector.h"
#include "preprocessing.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace hcce {

YoloDetector::YoloDetector(const std::string& model_path,
                             const std::vector<int>& obj_id_list,
                             const std::string& cuda_device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector")
    , obj_id_list_(obj_id_list)
{
    session_opts_.SetIntraOpNumThreads(4);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_opts;
    cuda_opts.device_id = std::stoi(cuda_device);
    session_opts_.AppendExecutionProvider_CUDA(cuda_opts);

    // OrtTensorRTProviderOptions trt_opts{};
    // trt_opts.device_id = std::stoi(cuda_device);
    // trt_opts.trt_engine_cache_enable = 1;
    // trt_opts.trt_engine_cache_path = "./trt_cache";
    // session_opts_.AppendExecutionProvider_TensorRT(trt_opts);
#endif

    std::wstring model_path_w(model_path.begin(), model_path.end());
    session_ = Ort::Session(env_, model_path_w.c_str(), session_opts_);
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& img_bgr,
                                               float conf_threshold,
                                               float iou_threshold,
                                               int   max_det)
{
    // ── 1. Preprocessing ────────────────────────────────────────────────────
    auto prep = Preprocessing::prepareForYolo(img_bgr);

    // BGR → RGB, uint8 → float32 [0,1], HWC → CHW
    cv::Mat rgb;
    cv::cvtColor(prep.img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat flt;
    rgb.convertTo(flt, CV_32F, 1.0 / 255.0);

    const int H = flt.rows, W = flt.cols;
    std::vector<float> input(3 * H * W);
    std::vector<cv::Mat> channels(3);
    cv::split(flt, channels);
    for (int c = 0; c < 3; c++)
        std::memcpy(input.data() + c * H * W, channels[c].data, H * W * sizeof(float));

    // ── 2. ONNX inference ────────────────────────────────────────────────────
    std::array<int64_t, 4> input_shape = {1, 3, H, W};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), input_shape.data(), 4);

    const char* input_names[]  = {"images"};
    const char* output_names[] = {"output0"};

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names,  &input_tensor, 1,
                                output_names, 1);

    // ── 3. Parsování výstupu ─────────────────────────────────────────────────
    // YOLO11 výstup: [1, 4 + num_classes, num_anchors]
    auto& out_tensor = outputs[0];
    auto shape = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    // shape = [1, 4+num_classes, num_anchors]
    int num_rows    = (int)shape[1]; // 4 + num_classes
    int num_anchors = (int)shape[2];
    int num_classes = num_rows - 4;

    const float* data = out_tensor.GetTensorData<float>();

    auto dets = parseYoloOutput(data, num_anchors, num_classes,
                                 img_bgr.cols, img_bgr.rows,
                                 conf_threshold,
                                 prep.scale, prep.pad_left, prep.pad_top);

    // ── 4. NMS + omezení max_det ─────────────────────────────────────────────
    auto result = nms(dets, iou_threshold);
    if ((int)result.size() > max_det)
        result.resize(max_det);
    return result;
}

std::vector<Detection> YoloDetector::parseYoloOutput(
    const float* data, int num_anchors, int num_classes,
    int orig_w, int orig_h,
    float conf_threshold,
    float scale, int pad_left, int pad_top)
{
    // data layout: [4+num_classes, num_anchors] (sloupcové pořadí po transpozici)
    // YOLO11 exportuje: řádky = atributy, sloupce = kotvy
    // bbox formát: cx, cy, w, h (normalizované na input rozlišení)

    std::vector<Detection> dets;
    const int num_rows = 4 + num_classes;

    // Pomocná funkce: přístup data[row * num_anchors + anchor]
    auto at = [&](int row, int anchor) -> float {
        return data[row * num_anchors + anchor];
    };

    for (int a = 0; a < num_anchors; a++) {
        // Najdi třídu s nejvyšší confidence
        int   best_cls  = 0;
        float best_conf = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            float conf = at(4 + c, a);
            if (conf > best_conf) {
                best_conf = conf;
                best_cls  = c;
            }
        }
        if (best_conf < conf_threshold) continue;

        // Bbox (cx, cy, w, h) ve vstupním rozlišení sítě
        float cx = at(0, a);
        float cy = at(1, a);
        float bw = at(2, a);
        float bh = at(3, a);

        // Zpět na původní rozlišení obrazu
        // 1. Odeber padding
        float x1 = (cx - bw / 2.0f - pad_left) * scale;
        float y1 = (cy - bh / 2.0f - pad_top)  * scale;
        float x2 = (cx + bw / 2.0f - pad_left)  * scale;
        float y2 = (cy + bh / 2.0f - pad_top)   * scale;

        // Clamp na rozměry obrazu
        x1 = std::max(0.0f, std::min(x1, (float)(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(orig_h - 1)));

        if (x2 <= x1 || y2 <= y1) continue;

        Detection det;
        det.bbox       = cv::Rect2f(x1, y1, (x2-x1), (y2-y1));
        det.confidence = best_conf;
        det.class_id   = best_cls;
        det.obj_id     = (best_cls < (int)obj_id_list_.size())
                         ? obj_id_list_[best_cls] : best_cls + 1;
        dets.push_back(det);
    }
    return dets;
}

// ─── NMS ──────────────────────────────────────────────────────────────────────
float YoloDetector::iou(const cv::Rect& a, const cv::Rect& b)
{
    int ix = std::max(a.x, b.x);
    int iy = std::max(a.y, b.y);
    int ix2 = std::min(a.x + a.width,  b.x + b.width);
    int iy2 = std::min(a.y + a.height, b.y + b.height);
    if (ix2 <= ix || iy2 <= iy) return 0.0f;
    float inter = (float)(ix2 - ix) * (iy2 - iy);
    float ua    = (float)(a.width * a.height + b.width * b.height) - inter;
    return inter / ua;
}

std::vector<Detection> YoloDetector::nms(std::vector<Detection>& dets,
                                           float iou_threshold)
{
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool>      suppressed(dets.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                iou(dets[i].bbox, dets[j].bbox) > iou_threshold)
                suppressed[j] = true;
        }
    }
    return result;
}

} // namespace hcce