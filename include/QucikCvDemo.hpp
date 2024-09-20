#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <opencv2/imgproc.hpp>

using namespace cv;

class PIC
{
public:
    void colorSpace(const Mat &);
    void show_blur(const Mat &);
    void paint(const Mat &);
    void ExtractFeature(const Mat &);
    void Simple_handle(const Mat &);
};