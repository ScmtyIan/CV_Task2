#include "QucikCvDemo.hpp"
#define mat Mat

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Mat image = imread("../resources/test_image.png");
    if (image.empty())
    {
        cerr << "Could not open or find the image,please check the path or image" << endl;
        return -1;
    }

    Mat img; // 缩小后的图片
    // resize(image, img, Size(400, 400));
    img = image;
    // imshow("origin_image", img);
    // PIC me;
    // // -----------------task1:hsv&&gray
    // me.colorSpace(img);
    // // -----------------task2:filter
    // me.show_blur(img);
    // //------------------task3:Extraction_of_Feature
    // me.ExtractFeature(img);
    // // ------------------task4:Paint 在task3中已经绘制出了红色外轮廓和bounding box
    // me.paint(img);
    // //--------------------task5:rotation and cut
    // me.Simple_handle(img);
    //
    waitKey();
    destroyAllWindows();

    return 0;
}