#include "QucikCvDemo.hpp"

#define iw imwrite
#define is imshow

using namespace cv;
using namespace std;

void PIC::colorSpace(const Mat &img)
{
    Mat gray, hsv, bgra;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cvtColor(img, hsv, COLOR_BGR2HSV);
    imshow("GRAY", gray);
    imshow("HSV", hsv);
    imwrite("../resources/dst/task1/gray.jpg", gray);
    imwrite("../resources/dst/task1/hsv.jpg", hsv);
}

void PIC::show_blur(const Mat &img)
{
    Mat gua, mean;
    GaussianBlur(img, gua, Size(7, 7), 100);
    blur(img, mean, Size(5, 5), Point(-1, -1));
    imshow("gua", gua);
    imshow("mean", mean);
    imwrite("../resources/dst/task2/gua.jpg", gua);
    imwrite("../resources/dst/task2/mean.jpg", mean);
}

void PIC::paint(const Mat &img)
{
    Mat rec(Size(200, 400), CV_8UC3, Scalar(255, 255, 255));
    rectangle(rec, Rect(10, 10, 50, 300), Scalar(255, 0, 0), 2, LINE_8);
    Point p1 = Point(130, 130);
    Point p2 = Point(160, 290);
    line(rec, p1, p2, Scalar(0, 255, 255), 2, LINE_AA);
    circle(rec, Point(120, 260), 30, Scalar(0, 0, 255), -1, LINE_4);
    putText(rec, "OrzOrzOrzOrz", Point(10, 390), FONT_HERSHEY_SIMPLEX, 1, Scalar(159, 45, 87), 3, LINE_MAX);
    is("rec", rec);
    iw("../resources/dst/task4/foundation.jpg", rec);
}

void PIC::ExtractFeature(const Mat &img)
{
    // 高斯滤波后面好算面积
    Mat img_blur = img.clone();
    GaussianBlur(img_blur, img_blur, Size(5, 5), 10);

    //--------------------------------------提取红色部分
    // 摆烂了，饱和度和亮度看情况拉个区间得了
    // 提取红色区域，red～～[0, 10]U[156,180]
    Mat red1, red2, hsv = img_blur.clone(), mask, red_zone;
    // 两区间的红色提取二值图像
    cvtColor(img_blur, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(156, 140, 140), Scalar(180, 255, 255), red1);
    inRange(hsv, Scalar(0, 120, 120), Scalar(10, 255, 255), red2);
    // 向家生成掩模mask并做and运算抽取原图红色部分
    add(red1, red2, mask, Mat(), CV_8U); // 改成位运算也可以
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    bitwise_and(mask, img, red_zone);
    imshow("red_zone", red_zone);
    imwrite("../resources/dst/task3/red_zone.jpg", red_zone);

    //-----------------------------------寻找红色外轮廓
    // 直接用掩模mask先消除内外噪点，再寻找外轮廓(主要是后面要算面积)
    cvtColor(mask, mask, COLOR_BGR2GRAY);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    morphologyEx(mask, mask, MORPH_OPEN, kernel, Point(-1, -1), 2);
    // 寻找外轮廓
    Mat paper = img.clone();
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(-2, -2)); // 两次开运算把轮廓搞偏了,调回来
    drawContours(paper, contours, -1, Scalar(10, 97, 166), 5, LINE_8);
    imshow("red_external_contours", paper);
    imwrite("../resources/dst/task3/red_external_contours.jpg", paper);

    //---------------------------寻找外轮廓的bounding box&& claculate AreaSize
    // 这里采用rect，也可以用凸包或circle
    paper = img.clone();
    // 排序目前没用
    sort(contours.begin(), contours.end(), [](const vector<Point> &a, const vector<Point> &b)
         { return cv::contourArea(a) < cv::contourArea(b); });
    // 遍历每个外轮廓，画出bounding box，并顺便计算外轮廓面积
    for (auto con : contours)
    {
        Rect x = boundingRect(con);
        int area = (int)contourArea(con);
        if (area < 1000) // 太小的不要了
            continue;
        polylines(paper, con, 1, Scalar(255, 0, 0), 1);                                                                                       // 绘制轮廓
        rectangle(paper, x, Scalar(45, 197, 166), 3, LINE_8);                                                                                 // 绘制boundingbox
        putText(paper, to_string(area), Point(x.x + x.width / 3, x.y + x.height / 2), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 0), 2, LINE_8); // 标注面积
    }
    imshow("red_bounding_box&&Size", paper);
    imwrite("../resources/dst/task3/red_bounding_box&&size.jpg", paper);

    //---------------------------------------------------高亮部分的处理
    // 提取高饱和高亮度的区域，用mask制作掩模
    inRange(hsv, Scalar(0, 160, 160), Scalar(180, 255, 255), mask);
    cvtColor(mask, mask, COLOR_GRAY2BGR);
    bitwise_and(mask, img, paper);
    imshow("highlight", paper);
    imwrite("../resources/dst/task3/highlight.jpg", paper);

    Mat dst;
    cvtColor(paper, dst, COLOR_BGR2GRAY);
    imshow("gray", dst);
    imwrite("../resources/dst/task3/dst_gray.jpg", dst);
    threshold(dst, dst, 70, 255, THRESH_BINARY);
    imshow("binary", dst);
    imwrite("../resources/dst/task3/dst_binary.jpg", dst);
    dilate(dst, dst, kernel, Point(-1, -1), 3);
    imshow("dilate", dst);
    imwrite("../resources/dst/task3/dst_dilate.jpg", dst);
    erode(dst, dst, kernel, Point(-1, -1), 7);
    imshow("erode", dst);
    imwrite("../resources/dst/task3/dst_erode.jpg", dst);
    // 漫水填充
    mask.create(img.rows + 2, img.cols + 2, CV_8UC1);
    mask = Scalar(0);
    Rect ccomp = Rect();
    floodFill(paper, mask, Point(img.cols / 2, img.rows / 2), Scalar(255, 255, 0), &ccomp, Scalar(50, 50, 50), Scalar(50, 50, 50));
    rectangle(paper, ccomp, Scalar(255, 0, 0), 3); // 绘制填充区最小外切矩形
    imshow("floodFill", paper);
    imwrite("../resources/dst/task3/dst_floodFill.jpg", paper);
}

void PIC::Simple_handle(const Mat &img)
{
    // 旋转逆时针35度
    int x, y;
    Mat paper;
    img.copyTo(paper);
    x = paper.cols;
    y = paper.rows;
    Mat m = getRotationMatrix2D(Point2f(x / 2, y / 2), 35, 1);
    warpAffine(paper, paper, m, Size());
    imshow("rotate", paper);
    imwrite("../resources/dst/task5/rotate.jpg", paper);
    // 裁减左上角1/4
    Mat quarter = img(Rect(0, 0, img.cols / 2, img.rows / 2));
    namedWindow(("quarter"), 1);
    resizeWindow("quarter", 400, 400);
    imshow(("quarter"), quarter);
    imwrite("../resources/dst/task5/quarter.jpg", quarter);
}