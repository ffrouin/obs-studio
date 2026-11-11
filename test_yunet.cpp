#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/face.hpp>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) return 3;
    auto detector = cv::FaceDetectorYN::create(argv[2], "", image.size(), 0.35f, 0.3f, 5000);
    if (detector.empty()) return 4;
    cv::Mat detections;
    detector->detect(image, detections);
    std::cout << "image=" << image.cols << "x" << image.rows
              << " rows=" << detections.rows << " cols=" << detections.cols << "\n";
    for (int row = 0; row < detections.rows; ++row) {
        std::cout << "row=" << row
                  << " x=" << detections.at<float>(row, 0)
                  << " y=" << detections.at<float>(row, 1)
                  << " w=" << detections.at<float>(row, 2)
                  << " h=" << detections.at<float>(row, 3)
                  << " score=" << detections.at<float>(row, 14) << "\n";
    }
    return 0;
}
