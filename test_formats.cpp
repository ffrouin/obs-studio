#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <iostream>
#include <vector>

static int detect_count(const cv::Mat &bgr, const std::string &model)
{
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(400, 272), 0, 0, cv::INTER_AREA);
    auto detector = cv::FaceDetectorYN::create(model, "", small.size(), 0.35f, 0.3f, 5000);
    cv::Mat out;
    detector->detect(small, out);
    return out.rows;
}

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    cv::Mat source = cv::imread(argv[1]);
    if (source.empty()) return 3;
    const std::string model = argv[2];

    cv::Mat bgra, rgba;
    cv::cvtColor(source, bgra, cv::COLOR_BGR2BGRA);
    cv::cvtColor(source, rgba, cv::COLOR_BGR2RGBA);

    cv::Mat from_bgra, from_rgba, from_bgra_flipped, from_rgba_flipped;
    cv::cvtColor(bgra, from_bgra, cv::COLOR_BGRA2BGR);
    cv::cvtColor(rgba, from_rgba, cv::COLOR_RGBA2BGR);
    cv::flip(from_bgra, from_bgra_flipped, 0);
    cv::flip(from_rgba, from_rgba_flipped, 0);

    std::cout << "original=" << detect_count(source, model) << "\n";
    std::cout << "BGRA->BGR=" << detect_count(from_bgra, model) << "\n";
    std::cout << "RGBA->BGR=" << detect_count(from_rgba, model) << "\n";
    std::cout << "flipped BGRA->BGR=" << detect_count(from_bgra_flipped, model) << "\n";
    std::cout << "flipped RGBA->BGR=" << detect_count(from_rgba_flipped, model) << "\n";
    return 0;
}
