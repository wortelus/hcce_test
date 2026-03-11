#include <windows.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "types.h"
#include "preprocessing.h"
#include "yolodetector.h"
#include "croptransform.h"
#include "hcceposeestimator.h"
#include "hccedecoder.h"
#include "ultradensesampler.h"
#include "pnpsolver.h"
#include "visualizer.h"

using namespace hcce;

// ─── Načtení models_info.json (zjednodušená verze bez JSON knihovny) ──────────
// Pro produkci doporučujeme nlohmann/json
struct ModelsInfo {
    std::map<int, ObjectInfo> objects;
};

// Pomocná funkce pro vytvoření 3D bbox rohů z ObjectInfo
std::vector<cv::Point3f> makeBBox3DCorners(const ObjectInfo &info) {
    float x0 = info.min_x, x1 = info.min_x + info.size_x;
    float y0 = info.min_y, y1 = info.min_y + info.size_y;
    float z0 = info.min_z, z1 = info.min_z + info.size_z;
    return {
        {x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0},
        {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}
    };
}

// ─── Hlavní třída (ekvivalent Python Tester) ──────────────────────────────────
class HccePoseTester {
public:
    struct Config {
        std::string dataset_path;
        std::string cuda_device = "0";
        int crop_size = 256;
        float conf_thresh = 0.85f;
        float iou_thresh = 0.50f;
        float mask_thresh = 0.0f;
        bool show_vis = true;
    };

    HccePoseTester(const Config &cfg,
                   const std::vector<int> &obj_id_list,
                   const std::vector<ObjectInfo> &obj_infos)
        : cfg_(cfg) {
        // Ulož informace o objektech
        for (size_t i = 0; i < obj_id_list.size(); i++) {
            obj_id_list_.push_back(obj_id_list[i]);
            obj_infos_[obj_id_list[i]] = obj_infos[i];
        }

        // ── YOLO detektor ────────────────────────────────────────────────────
        std::string yolo_path = cfg.dataset_path
                                + "/yolo11-detection-obj_s.onnx";
        yolo_ = std::make_unique<YoloDetector>(yolo_path, obj_id_list, cfg.cuda_device);

        // ── HccePose modely (jeden na objekt) ────────────────────────────────
        for (int obj_id: obj_id_list) {
            std::string id_str = std::to_string(obj_id);
            while ((int) id_str.size() < 2) id_str = "0" + id_str;

            std::string model_path = cfg.dataset_path
                                     + "/hccepose_obj_" + std::to_string(obj_id) + ".onnx";

            auto &info = obj_infos_[obj_id];
            HcceDecoder::ObjBounds bounds{
                info.min_x, info.min_y, info.min_z,
                info.size_x, info.size_y, info.size_z
            };

            estimators_[obj_id] = std::make_unique<HccePoseEstimator>(
                model_path, cfg.cuda_device);
            decoders_[obj_id] = std::make_unique<HcceDecoder>(bounds);
        }

        pnp_solver_ = std::make_unique<PnPSolver>();
        std::cout << "[HccePoseTester] Inicializace dokončena.\n";
    }

    std::vector<float> loadNpy(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        // Přeskoč npy header
        char magic[6]; f.read(magic, 6);
        uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
        uint16_t header_len; f.read((char*)&header_len, 2);
        std::vector<char> header(header_len);
        f.read(header.data(), header_len);
        // Načti data
        std::vector<float> data(3 * 256 * 256);
        f.read((char*)data.data(), data.size() * sizeof(float));
        return data;
    }

    // ─── Predikce 6D pózy pro jeden obraz ────────────────────────────────────
    struct PredictResult {
        std::vector<PoseResult> poses;
        cv::Mat vis_2d;
        cv::Mat vis_6d;
        double elapsed_ms;
    };

    PredictResult predict(const cv::Mat &img_bgr,
                          const CameraIntrinsics &cam,
                          const std::vector<int> &target_obj_ids) {
        auto t0 = std::chrono::high_resolution_clock::now();

        PredictResult out;
        cv::Mat K = cam.toMat();

        // ── 0. Downscale + padding (stejně jako Python) ──────────────────────────
        float ratio = std::max((float)img_bgr.rows / 640.0f,
                               (float)img_bgr.cols / 640.0f);
        if (ratio < 1.0f) ratio = 1.0f;

        cv::Mat img_scaled;
        if (ratio > 1.0f) {
            int H_new = (int)(img_bgr.rows / ratio);
            int W_new = (int)(img_bgr.cols / ratio);
            cv::resize(img_bgr, img_scaled, cv::Size(W_new, H_new), 0, 0, cv::INTER_LINEAR);
            K.at<double>(0,0) /= ratio;  // fx
            K.at<double>(1,1) /= ratio;  // fy
            K.at<double>(0,2) /= ratio;  // cx
            K.at<double>(1,2) /= ratio;  // cy
        } else {
            img_scaled = img_bgr.clone();
        }

        // Padding na násobek 32 (doprava a dolů, stejně jako Python)
        int H = img_scaled.rows, W = img_scaled.cols;
        int H_pad = (H % 32 == 0) ? H : (H / 32 + 1) * 32;
        int W_pad = (W % 32 == 0) ? W : (W / 32 + 1) * 32;
        int h_move = (H_pad - H) / 2;
        int w_move = (W_pad - W) / 2;
        if (h_move > 0 || w_move > 0) {
            cv::Mat padded = cv::Mat::zeros(H_pad, W_pad, img_scaled.type());
            img_scaled.copyTo(padded(cv::Rect(w_move, h_move, W, H)));
            img_scaled = padded;
            K.at<double>(0,2) += w_move;
            K.at<double>(1,2) += h_move;
        }

        // ── 1. YOLO detekce ──────────────────────────────────────────────────
        auto t_detection_start = std::chrono::high_resolution_clock::now();
        auto detections = yolo_->detect(img_scaled, cfg_.conf_thresh, cfg_.iou_thresh);
        auto t_detection_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> detection_time = t_detection_end - t_detection_start;
        std::cout << "[HccePoseTester] Detekce dokončena: " << detections.size() << " objektů, čas = "
                << detection_time.count() << " ms\n";

        // ── 2. Pro každý detekovaný objekt (který nás zajímá) ────────────────
        cv::Mat vis = img_scaled.clone();

        cv::Mat img_float_full;
        img_scaled.convertTo(img_float_full, CV_32FC3, 1.0 / 255.0);
        cv::Mat ch_full[3];
        cv::split(img_float_full, ch_full);
        ch_full[0] = (ch_full[0] - 0.485f) / 0.229f;
        ch_full[1] = (ch_full[1] - 0.456f) / 0.224f;
        ch_full[2] = (ch_full[2] - 0.406f) / 0.225f;
        cv::merge(ch_full, 3, img_float_full);

        for (const auto &det: detections) {
            // Přeskočit pokud objekt není v cílovém seznamu
            if (std::find(target_obj_ids.begin(), target_obj_ids.end(),
                          det.obj_id) == target_obj_ids.end())
                continue;

            int obj_id = det.obj_id;
            if (estimators_.find(obj_id) == estimators_.end()) continue;

            // ── 3. Crop transformace ─────────────────────────────────────────
            // Škáluj bbox do downscalovaného prostoru
            auto crop_info_256 = CropTransform::compute(det.bbox, 256, 1.5f);
            auto crop_info_128 = CropTransform::computeHalf(det.bbox, 1.5f);

            // 256×256 pro síť
            auto t_crop_start = std::chrono::high_resolution_clock::now();
            // auto crop_info_256 = CropTransform::compute(det.bbox, 256, 1.5f);
            // Převeď celý obraz na float RGB a normalizuj PŘED cropem

            // Crop na nenormalizovaném float obraze
            cv::Mat crop_raw = CropTransform::warp(img_scaled, crop_info_256, 256);

            // 2. Pak cropni z normalizovaného float obrazu
            cv::Mat crop = CropTransform::warp(img_float_full, crop_info_256, 256);

            // === DEBUG: porovnání s Pythonem ===
            auto chw_cpp = Preprocessing::floatMatToCHW(crop);
            int idx = 128 * 256 + 128;
            printf("[CPP] crop CHW pixel[128,128] ch0=%.6f ch1=%.6f ch2=%.6f\n",
                   chw_cpp[idx], chw_cpp[256*256 + idx], chw_cpp[2*256*256 + idx]);

            // Bbox a crop info:
            printf("[CPP] bbox: x=%.2f y=%.2f w=%.2f h=%.2f\n",
                   det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
            printf("[CPP] crop_info_256 M:\n");
            for (int i = 0; i < 3; i++)
                printf("  [%.6f %.6f %.6f]\n",
                       crop_info_256.M.at<double>(i,0),
                       crop_info_256.M.at<double>(i,1),
                       crop_info_256.M.at<double>(i,2));

            printf("[CPP] h_move=%d w_move=%d img_scaled shape: %dx%d\n", h_move, w_move, img_scaled.cols, img_scaled.rows);

            // 128×128 pro zpětnou transformaci výstupu sítě
            // auto crop_info_128 = CropTransform::computeHalf(det.bbox, 1.5f);
            auto t_crop_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> crop_time = t_crop_end - t_crop_start;
            std::cout << "[HccePoseTester] Crop transformace dokončena, čas = "
                    << crop_time.count() << " ms\n";

            // ── 4. HccePose inference ─────────────────────────────────────────
            // HWC→CHW bez normalizace
            auto t_hcce_inference_start = std::chrono::high_resolution_clock::now();
            // auto chw = Preprocessing::prepareForHccePose(crop);

            auto chw = Preprocessing::floatMatToCHW(crop);
            auto net_out = estimators_[obj_id]->infer(chw);

            // auto chw = loadNpy("debug_crop_for_cpp.npy");
            // auto net_out = estimators_[obj_id]->infer(chw);

            auto t_hcce_inference_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> hcce_inference_time =
                    t_hcce_inference_end - t_hcce_inference_start;
            std::cout << "[HccePoseTester] HccePose inference dokončena, čas = "
                    << hcce_inference_time.count() << " ms\n";

            printf("[CPP DEBUG] YOLO bbox: x=%.2f y=%.2f w=%.2f h=%.2f\n",
            det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
            printf("[CPP DEBUG] crop_info_128 M:\n");
            for (int i = 0; i < 3; i++)
                printf("  [%.6f %.6f %.6f]\n",
                       crop_info_128.M.at<double>(i,0),
                       crop_info_128.M.at<double>(i,1),
                       crop_info_128.M.at<double>(i,2));

            // -- Debug: ulož crop
            // Debug: ulož crop - de-normalizuj zpět na uint8
            cv::Mat crop_vis = crop.clone();
            cv::Mat dchannels[3];
            cv::split(crop_vis, dchannels);
            dchannels[0] = dchannels[0] * 0.229f + 0.485f;
            dchannels[1] = dchannels[1] * 0.224f + 0.456f;
            dchannels[2] = dchannels[2] * 0.225f + 0.406f;
            cv::merge(dchannels, 3, crop_vis);
            crop_vis *= 255.0f;
            crop_vis.convertTo(crop_vis, CV_8UC3);
            cv::cvtColor(crop_vis, crop_vis, cv::COLOR_RGB2BGR);
            cv::imwrite("./test_imgs/debug_crop_obj" + std::to_string(obj_id) + ".jpg", crop_vis);

            // ── 5. HCCE dekódování → 3D souřadnice ───────────────────────────
            auto t_decode_start = std::chrono::high_resolution_clock::now();
            auto decoded = decoders_[obj_id]->decode(net_out, cfg_.mask_thresh);
            std::cerr << "[decode] pixels_2d.size()=" << decoded.pixels_2d.size() << "\n";

            // === DEBUG ===
            if (!decoded.pixels_2d.empty()) {
                // Najdi pixel nejbližší k (68, 61)
                int target_r = 61, target_c = 68;
                int best_idx = -1;
                float best_dist = 1e9f;
                for (size_t i = 0; i < decoded.pixels_2d.size(); i++) {
                    float dr = decoded.pixels_2d[i].y - target_r;
                    float dc = decoded.pixels_2d[i].x - target_c;
                    float d = dr*dr + dc*dc;
                    if (d < best_dist) { best_dist = d; best_idx = i; }
                }
                int r = (int)decoded.pixels_2d[best_idx].y;
                int c = (int)decoded.pixels_2d[best_idx].x;
                printf("[CPP DEBUG] pixel=(%d,%d)\n", c, r);
                const float* front = net_out.front_codes.ptr<float>(r) + c * 24;
                printf("[CPP DEBUG] raw front codes: ");
                for (int j = 0; j < 24; j++) printf("%.6f ", front[j]);
                printf("\n");
                printf("[CPP DEBUG] front_3d: %.4f %.4f %.4f\n",
                       decoded.front_3d[best_idx].x,
                       decoded.front_3d[best_idx].y,
                       decoded.front_3d[best_idx].z);
                auto pt = CropTransform::transformPoint(
                    decoded.pixels_2d[best_idx], crop_info_128.M_inv);
                printf("[CPP DEBUG] coord_2d: %.4f %.4f\n", pt.x, pt.y);
            }

            // DEBUG: vizualizace front 3D souřadnic jako RGB
            if (cfg_.show_vis) {
                // Nejdřív projdi všechny pixely a najdi min/max
                float min_x = 1.0f, max_x = 0.0f;
                float min_y = 1.0f, max_y = 0.0f;
                float min_z = 1.0f, max_z = 0.0f;

                for (size_t i = 0; i < decoded.pixels_2d.size(); i++) {
                    // Normalizované souřadnice (před de-normalizací)
                    int r = (int) decoded.pixels_2d[i].y;
                    int c = (int) decoded.pixels_2d[i].x;
                    const float *front = net_out.front_codes.ptr<float>(r) + c * 24;
                    float nx = HcceDecoder::decodeComponent(front + 0);
                    float ny = HcceDecoder::decodeComponent(front + 8);
                    float nz = HcceDecoder::decodeComponent(front + 16);
                    min_x = std::min(min_x, nx);
                    max_x = std::max(max_x, nx);
                    min_y = std::min(min_y, ny);
                    max_y = std::max(max_y, ny);
                    min_z = std::min(min_z, nz);
                    max_z = std::max(max_z, nz);
                }

                // Pak vizualizuj s normalizací na celý rozsah
                cv::Mat coord_vis(128, 128, CV_8UC3, cv::Scalar(0, 0, 0));
                for (size_t i = 0; i < decoded.pixels_2d.size(); i++) {
                    int r = (int) decoded.pixels_2d[i].y;
                    int c = (int) decoded.pixels_2d[i].x;
                    const float *front = net_out.front_codes.ptr<float>(r) + c * 24;
                    float nx = HcceDecoder::decodeComponent(front + 0);
                    float ny = HcceDecoder::decodeComponent(front + 8);
                    float nz = HcceDecoder::decodeComponent(front + 16);
                    coord_vis.at<cv::Vec3b>(r, c) = cv::Vec3b(
                        (uchar) ((nx - min_x) / (max_x - min_x + 1e-6f) * 255),
                        (uchar) ((ny - min_y) / (max_y - min_y + 1e-6f) * 255),
                        (uchar) ((nz - min_z) / (max_z - min_z + 1e-6f) * 255)
                    );
                }
                std::string coord_path = "./test_imgs/debug_coords_obj"
                                         + std::to_string(obj_id) + ".jpg";
                cv::imwrite(coord_path, coord_vis);
                std::cerr << "Coord viz uložena: " << coord_path << "\n";
            }

            if (decoded.pixels_2d.empty()) {
                std::cerr << "[warn] obj_" << obj_id << ": prázdná maska\n";
                continue;
            }

            auto t_decode_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> decode_time = t_decode_end - t_decode_start;
            std::cout << "[HccePoseTester] Dekódování dokončeno, čas = "
                    << decode_time.count() << " ms\n";

            // ── 6. Ultra-dense sampling ───────────────────────────────────────
            auto t_sampling_start = std::chrono::high_resolution_clock::now();
            auto dense_corr = UltraDenseSampler::sample(
                decoded.pixels_2d, decoded.front_3d, decoded.back_3d);
            auto t_sampling_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> sampling_time = t_sampling_end - t_sampling_start;
            std::cout << "[HccePoseTester] Ultra-dense sampling dokončen, čas = "
                    << sampling_time.count() << " ms\n";

            // ── 7. RANSAC-PnP ──────────────────────────────────────────────────
            auto t_pnp_start = std::chrono::high_resolution_clock::now();
            PoseResult pose = pnp_solver_->solve(
                decoded.front_3d,
                decoded.back_3d,
                decoded.pixels_2d,
                K,
                crop_info_128);

            printf("[PnP] N bodů: %d\n", static_cast<int>(dense_corr.pts3d.size()));
            pose.obj_id = obj_id;
            pose.confidence = det.confidence;

            if (pose.valid) {
                out.poses.push_back(pose);

                // ── 8. Vizualizace ─────────────────────────────────────────
                if (cfg_.show_vis) {
                    Visualizer::draw3DBBox(vis, pose.R, pose.t, K,
                                           obj_infos_[obj_id]);
                    Visualizer::drawAxes(vis, pose.R, pose.t, K, 30.0f);
                }
            }
            auto t_pnp_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> pnp_time = t_pnp_end - t_pnp_start;
            std::cout << "[HccePoseTester] RANSAC-PnP dokončen, čas = "
                    << pnp_time.count() << " ms\n";
        }



        if (cfg_.show_vis) {
            Visualizer::drawDetections(vis, detections);
            out.vis_6d = vis;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return out;
    }

private:
    Config cfg_;
    std::vector<int> obj_id_list_;
    std::map<int, ObjectInfo> obj_infos_;
    std::unique_ptr<YoloDetector> yolo_;
    std::map<int, std::unique_ptr<HccePoseEstimator> > estimators_;
    std::map<int, std::unique_ptr<HcceDecoder> > decoders_;
    std::unique_ptr<PnPSolver> pnp_solver_;
};

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    SetConsoleOutputCP(CP_UTF8);

    if (argc < 2) {
        std::cerr << "Usage: hccepose <dataset_path> [image_path]\n";
        return 1;
    }

    // Zkontroluj zda soubory existují
    std::string yolo_path = std::string(argv[1])
                            + "/yolo11-detection-obj_s.onnx";

    std::ifstream f(yolo_path);
    if (!f.good()) {
        std::cerr << "ONNX model nenalezen: " << yolo_path << "\n";
        return 1;
    }

    std::cerr << "Model nalezen OK\n";


    try {
        // ── Konfigurace ────────────────────────────────────────────────────────
        HccePoseTester::Config cfg;
        cfg.dataset_path = argv[1];
        cfg.cuda_device = "0";
        cfg.conf_thresh = 0.85f;
        cfg.show_vis = true;

        // ── Definice objektů (odpovídá models_info.json) ───────────────────────
        // V produkci načítej ze souboru JSON
        std::vector<int> obj_ids = {1};
        std::vector<ObjectInfo> obj_infos = {
            {
                1,
                -70.50000762939453, -134.61109924316406, -46.094966888427734, // min_x, min_y, min_z [mm]
                141.0, 269.2908935546875, 92.45808410644531, // size_x, size_y, size_z [mm]
                {}
            }
        };
        obj_infos[0].bbox3d_corners = makeBBox3DCorners(obj_infos[0]);

        // ── Parametry kamery ────────────────────────────────────────────────────
        CameraIntrinsics cam;
        // zed 2i
        cam.fx = 1106.4165817481885;
        cam.fy = 1106.4165817481885;
        cam.cx = 670.8988620923914;
        cam.cy = 372.0583177649457;

        // ── Inicializace ────────────────────────────────────────────────────────
        HccePoseTester tester(cfg, obj_ids, obj_infos);

        // ── Zpracování obrazu nebo videa ────────────────────────────────────────
        std::string input_path = (argc >= 3) ? argv[2] : "";

        if (input_path.empty()) {
            std::cerr << "Zadej cestu k obrazu nebo videu.\n";
            return 1;
        }

        // ── Detekce formátu ─────────────────────────────────────────────────────
        std::string ext = input_path.substr(input_path.rfind('.'));
        bool is_video = (ext == ".mp4" || ext == ".avi" || ext == ".mov");

        if (!is_video) {
            // ── Jeden obraz ─────────────────────────────────────────────────────
            cv::Mat img = cv::imread(input_path);
            if (img.empty()) {
                std::cerr << "Nelze načíst obraz: " << input_path << "\n";
                return 1;
            }

            cv::Mat img_swapped;
            cv::cvtColor(img, img_swapped, cv::COLOR_BGR2RGB);
            auto result = tester.predict(img_swapped, cam, obj_ids);

            std::cout << "Zpracováno za " << result.elapsed_ms << " ms\n";
            std::cout << "Nalezeno " << result.poses.size() << " objektů\n";

            for (const auto &pose: result.poses) {
                std::cout << "\nobj_" << pose.obj_id << ":\n";
                std::cout << "  R = " << pose.R << "\n";
                std::cout << "  t = " << pose.t.t() << "\n";
            }

            if (!result.vis_6d.empty()) {
                std::string out_path = input_path.substr(0, input_path.rfind('.')) + "_6d.jpg";
                cv::imwrite(out_path, result.vis_6d);
                std::cout << "Uloženo: " << out_path << "\n";
            }
        } else {
            // ── Video ────────────────────────────────────────────────────────────
            cv::VideoCapture cap(input_path);
            if (!cap.isOpened()) {
                std::cerr << "Nelze otevřít video: " << input_path << "\n";
                return 1;
            }

            int W = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int H = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);

            std::string out_path = input_path.substr(0, input_path.rfind('.')) + "_6d.mp4";

            cv::VideoWriter writer;
            int frame_idx = 0;
            cv::Mat frame;
            while (cap.read(frame)) {
                auto result = tester.predict(frame, cam, obj_ids);

                double fps_pose = 1000.0 / result.elapsed_ms;
                std::cout << "\rFrame " << ++frame_idx
                        << "  FPS: " << std::fixed << std::setprecision(1) << fps_pose
                        << "  Objekty: " << result.poses.size()
                        << std::endl;

                if (!result.vis_6d.empty()) {
                    if (!writer.isOpened()) {
                        writer.open(out_path,
                                    cv::VideoWriter::fourcc('m','p','4','v'),
                                    fps,
                                    cv::Size(result.vis_6d.cols, result.vis_6d.rows));
                    }
                    cv::putText(result.vis_6d,
                                "FPS: " + std::to_string((int)fps_pose),
                                cv::Point(20, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 2.0,
                                cv::Scalar(0, 255, 0), 4);
                    writer.write(result.vis_6d);
                }

                // break;
            }
            std::cout << "\nUloženo: " << out_path << "\n";
            cap.release();
            writer.release();
        }

        return 0;
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime chyba: " << e.what() << "\n";
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Chyba: " << e.what() << "\n";
        return 1;
    }
}
