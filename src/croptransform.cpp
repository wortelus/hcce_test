#include "croptransform.h"
#include <cmath>

namespace hcce {

CropTransform::CropInfo CropTransform::compute(const cv::Rect2f& bbox,
                                                 int out_size,
                                                 float pad_ratio)
{
    // Centrum bbox
    float cx = bbox.x + bbox.width  / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;

    // Radius = max(w,h)/2 * padding_ratio
    float radius = std::max(bbox.width, bbox.height) * pad_ratio / 2.0f;

    float left   = std::round(cx - radius);
    float right  = std::round(cx + radius);
    float top    = std::round(cy - radius);
    float bottom = std::round(cy + radius);

    float side = right - left;  // = bottom - top (čtverec)

    // Homografie: orig → crop_out_size
    // Krok 1: posun (odeber left, top)
    // Krok 2: scale na out_size
    // Výsledná matice = Scale @ Translate
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    T.at<double>(0, 2) = -left;
    T.at<double>(1, 2) = -top;

    cv::Mat S = cv::Mat::eye(3, 3, CV_64F);
    S.at<double>(0, 0) = out_size / side;
    S.at<double>(1, 1) = out_size / side;

    CropInfo info;
    info.M     = S * T;
    info.M_inv = info.M.inv();
    info.src_bbox = cv::Rect((int)left, (int)top,
                              (int)(right - left), (int)(bottom - top));
    return info;
}

CropTransform::CropInfo CropTransform::computeHalf(const cv::Rect2f& bbox,
                                                      float pad_ratio)
{
    return compute(bbox, 128, pad_ratio);
}

cv::Mat CropTransform::warp(const cv::Mat& img, const CropInfo& info, int out_size)
{
    cv::Mat result;
    // warpPerspective aplikuje transformaci M na img
    cv::warpPerspective(img, result, info.M,
                         cv::Size(out_size, out_size),
                         cv::INTER_LINEAR,
                         cv::BORDER_CONSTANT,
                        cv::Scalar(0.0f, 0.0f, 0.0f));
    return result;
}

cv::Mat CropTransform::warpBack(const cv::Mat& crop_output,
                                  const CropInfo& info_128,
                                  int orig_h, int orig_w,
                                  int interp)
{
    cv::Mat result;
    cv::warpPerspective(crop_output, result, info_128.M_inv,
                         cv::Size(orig_w, orig_h),
                         interp,
                         cv::BORDER_CONSTANT,
                         cv::Scalar(0.0f, 0.0f, 0.0f));
    return result;
}

cv::Point2f CropTransform::transformPoint(const cv::Point2f& pt, const cv::Mat& M_inv)
{
    cv::Mat p = (cv::Mat_<double>(3,1) << pt.x, pt.y, 1.0);
    cv::Mat q = M_inv * p;
    return cv::Point2f((float)(q.at<double>(0) / q.at<double>(2)),
                       (float)(q.at<double>(1) / q.at<double>(2)));
}

} // namespace hcce