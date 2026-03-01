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

        // ── 1. YOLO detekce ──────────────────────────────────────────────────
        auto detections = yolo_->detect(img_bgr, cfg_.conf_thresh, cfg_.iou_thresh);

        // ── 2. Pro každý detekovaný objekt (který nás zajímá) ────────────────
        cv::Mat vis = img_bgr.clone();

        for (const auto &det: detections) {
            // Přeskočit pokud objekt není v cílovém seznamu
            if (std::find(target_obj_ids.begin(), target_obj_ids.end(),
                          det.obj_id) == target_obj_ids.end())
                continue;

            int obj_id = det.obj_id;
            if (estimators_.find(obj_id) == estimators_.end()) continue;

            // ── 3. Crop transformace ─────────────────────────────────────────
            // 256×256 pro síť
            auto crop_info_256 = CropTransform::compute(det.bbox, 256, 1.5f);
            // Převeď celý obraz na float RGB a normalizuj PŘED cropem
            cv::Mat img_float;
            cv::Mat img_rgb;
            cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
            img_rgb.convertTo(img_float, CV_32FC3, 1.0/255.0);

            // Normalizace
            cv::Mat channels[3];
            cv::split(img_float, channels);
            channels[0] = (channels[0] - 0.485f) / 0.229f;
            channels[1] = (channels[1] - 0.456f) / 0.224f;
            channels[2] = (channels[2] - 0.406f) / 0.225f;
            cv::merge(channels, 3, img_float);

            // Crop na normalizovaném float obraze
            cv::Mat crop = CropTransform::warp(img_float, crop_info_256, 256);

            // 128×128 pro zpětnou transformaci výstupu sítě
            auto crop_info_128 = CropTransform::computeHalf(det.bbox, 1.5f);

            // ── 4. HccePose inference ─────────────────────────────────────────
            // auto chw = Preprocessing::prepareForHccePose(crop);
            // HWC→CHW bez normalizace
            auto chw = Preprocessing::floatMatToCHW(crop);
            auto net_out = estimators_[obj_id]->infer(chw);

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
            auto decoded = decoders_[obj_id]->decode(net_out, cfg_.mask_thresh);

            // DEBUG: vizualizace front 3D souřadnic jako RGB
            if (cfg_.show_vis) {
                // Nejdřív projdi všechny pixely a najdi min/max
                float min_x = 1.0f, max_x = 0.0f;
                float min_y = 1.0f, max_y = 0.0f;
                float min_z = 1.0f, max_z = 0.0f;

                for (size_t i = 0; i < decoded.pixels_2d.size(); i++) {
                    // Normalizované souřadnice (před de-normalizací)
                    int r = (int)decoded.pixels_2d[i].y;
                    int c = (int)decoded.pixels_2d[i].x;
                    const float* front = net_out.front_codes.ptr<float>(r) + c * 24;
                    float nx = HcceDecoder::decodeComponent(front + 0);
                    float ny = HcceDecoder::decodeComponent(front + 8);
                    float nz = HcceDecoder::decodeComponent(front + 16);
                    min_x = std::min(min_x, nx); max_x = std::max(max_x, nx);
                    min_y = std::min(min_y, ny); max_y = std::max(max_y, ny);
                    min_z = std::min(min_z, nz); max_z = std::max(max_z, nz);
                }

                // Pak vizualizuj s normalizací na celý rozsah
                cv::Mat coord_vis(128, 128, CV_8UC3, cv::Scalar(0,0,0));
                for (size_t i = 0; i < decoded.pixels_2d.size(); i++) {
                    int r = (int)decoded.pixels_2d[i].y;
                    int c = (int)decoded.pixels_2d[i].x;
                    const float* front = net_out.front_codes.ptr<float>(r) + c * 24;
                    float nx = HcceDecoder::decodeComponent(front + 0);
                    float ny = HcceDecoder::decodeComponent(front + 8);
                    float nz = HcceDecoder::decodeComponent(front + 16);
                    coord_vis.at<cv::Vec3b>(r, c) = cv::Vec3b(
                        (uchar)((nx - min_x) / (max_x - min_x + 1e-6f) * 255),
                        (uchar)((ny - min_y) / (max_y - min_y + 1e-6f) * 255),
                        (uchar)((nz - min_z) / (max_z - min_z + 1e-6f) * 255)
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

            // ── 6. Ultra-dense sampling ───────────────────────────────────────
            auto dense_corr = UltraDenseSampler::sample(
                decoded.pixels_2d, decoded.front_3d, decoded.back_3d);

            // ── 7. RANSAC-PnP ──────────────────────────────────────────────────
            PoseResult pose = pnp_solver_->solve(dense_corr, K, crop_info_128);
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
        cam.fx = 3200.0;
        cam.fy = 3200.0;
        cam.cx = 1536.0;
        cam.cy = 864.0;

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

            auto result = tester.predict(img, cam, obj_ids);

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
            cv::VideoWriter writer(out_path,
                                   cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                   fps, cv::Size(W, H));

            int frame_idx = 0;
            cv::Mat frame;
            while (cap.read(frame)) {
                auto result = tester.predict(frame, cam, obj_ids);

                double fps_pose = 1000.0 / result.elapsed_ms;
                std::cout << "\rFrame " << ++frame_idx
                        << "  FPS: " << std::fixed << std::setprecision(1) << fps_pose
                        << "  Objekty: " << result.poses.size()
                        << std::flush;

                if (!result.vis_6d.empty()) {
                    cv::putText(result.vis_6d,
                                "FPS: " + std::to_string((int) fps_pose),
                                cv::Point(20, 60),
                                cv::FONT_HERSHEY_SIMPLEX, 2.0,
                                cv::Scalar(0, 255, 0), 4);
                    writer.write(result.vis_6d);
                }
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
