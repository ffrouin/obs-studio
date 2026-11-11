#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/face.hpp>
#include <iostream>
int main(int argc,char**argv){cv::Mat im=cv::imread(argv[1]); if(im.empty()) return 2; cv::resize(im,im,cv::Size(960,540)); auto d=cv::FaceDetectorYN::create(argv[2],"",cv::Size(320,320),0.35f,0.3f,5000); d->setInputSize(im.size()); cv::Mat o; d->detect(im,o); std::cout<<"rows="<<o.rows<<" cols="<<o.cols<<"\n"; for(int r=0;r<o.rows;r++){for(int c=0;c<15;c++)std::cout<<o.at<float>(r,c)<<(c==14?'\n':' ');} }
