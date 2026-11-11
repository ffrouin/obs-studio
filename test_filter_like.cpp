#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

struct Box { float x, y, w, h; };

static std::vector<Box> detect_like_filter(const cv::Mat &source, const std::string &model)
{
    cv::Mat source_bgra, bgr_source;
    cv::cvtColor(source, source_bgra, cv::COLOR_BGR2BGRA);
    cv::cvtColor(source_bgra, bgr_source, cv::COLOR_BGRA2BGR);
    const int width = std::max(160, static_cast<int>(bgr_source.cols * 0.5));
    const int height = std::max(90, static_cast<int>(bgr_source.rows * 0.5));
    cv::Mat bgr;
    cv::resize(bgr_source, bgr, cv::Size(width, height), 0, 0, cv::INTER_AREA);

    auto detector = cv::FaceDetectorYN::create(model, "", bgr.size(), 0.35f, 0.3f, 5000);
    cv::Mat detected;
    detector->detect(bgr, detected);
    std::vector<Box> result;
    for (int row = 0; row < detected.rows; ++row) {
        const float x = detected.at<float>(row, 0);
        const float y = detected.at<float>(row, 1);
        const float w = detected.at<float>(row, 2);
        const float h = detected.at<float>(row, 3);
        const float score = detected.at<float>(row, 14);
        if (score < 0.35f)
            continue;
        const float nx = std::clamp(x / bgr.cols, 0.0f, 1.0f);
        const float ny = std::clamp(y / bgr.rows, 0.0f, 1.0f);
        const float nw = std::clamp(w / bgr.cols, 0.0f, 1.0f - nx);
        const float nh = std::clamp(h / bgr.rows, 0.0f, 1.0f - ny);
        if (nw <= 0 || nh <= 0)
            continue;
        result.push_back({nx, ny, nw, nh});
    }
    return result;
}

static void draw_boxes(cv::Mat image, const std::vector<Box> &boxes, const std::string &path)
{
    for (const Box &b : boxes) {
        const int x = static_cast<int>(b.x * image.cols);
        const int y = static_cast<int>(b.y * image.rows);
        const int w = static_cast<int>(b.w * image.cols);
        const int h = static_cast<int>(b.h * image.rows);
        cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 3);
    }
    cv::imwrite(path, image);
}

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    const std::string image_path = argv[1];
    const std::string model_path = argv[2];
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) return 3;

    const auto boxes = detect_like_filter(image, model_path);
    std::cout << "filter-like detections=" << boxes.size() << "\n";
    for (const Box &b : boxes)
        std::cout << "analysis=" << b.x << "," << b.y << "," << b.w << "," << b.h
                  << " display_y=" << (1.0f - (b.y + b.h)) << "\n";
    draw_boxes(image, boxes, "test-images/filter_like_annotated.jpg");
    std::vector<Box> limited(boxes.begin(), boxes.begin() + std::min<size_t>(8, boxes.size()));
    draw_boxes(cv::imread(image_path), limited, "test-images/filter_like_annotated_max8.jpg");

    cv::Mat white(image.size(), image.type(), cv::Scalar(255, 255, 255));
    std::cout << "white detections=" << detect_like_filter(white, model_path).size() << "\n";
    return boxes.empty() ? 4 : 0;
}
