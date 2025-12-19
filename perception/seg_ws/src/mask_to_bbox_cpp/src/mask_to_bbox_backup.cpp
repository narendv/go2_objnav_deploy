#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <map>

// Custom comparator for cv::Vec3b
struct Vec3bCompare {
    bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const {
        return std::tie(lhs[0], lhs[1], lhs[2]) < std::tie(rhs[0], rhs[1], rhs[2]);
    }
};

// Define the color to object mapping
std::map<cv::Vec3b, std::string, Vec3bCompare> color_to_object = {
    {cv::Vec3b(204, 5, 255), "bed"},
    {cv::Vec3b(230, 230, 230), "window"},
    {cv::Vec3b(224, 5, 255), "cabinet"},
    {cv::Vec3b(8, 255, 51), "door"},
    {cv::Vec3b(255, 6, 82), "table"},
    {cv::Vec3b(255, 51, 7), "curtain"},
    {cv::Vec3b(204, 70, 3), "chair"},
    {cv::Vec3b(11, 102, 255), "sofa"},
    {cv::Vec3b(255, 7, 71), "shelf"},
    {cv::Vec3b(10, 255, 71), "table"},
    {cv::Vec3b(0, 255, 20), "box"},
    {cv::Vec3b(7, 255, 255), "closet"},
    {cv::Vec3b(255, 5, 153), "signboard"},
    {cv::Vec3b(6, 51, 255), "drawers"},
    {cv::Vec3b(0, 163, 255), "sink"},
    {cv::Vec3b(20, 255, 0), "refrigerator"},
    {cv::Vec3b(255, 71, 0), "table"},
    {cv::Vec3b(0, 173, 255), "door"},
    {cv::Vec3b(0, 255, 245), "bookcase"},
    {cv::Vec3b(0, 255, 112), "table"},
    {cv::Vec3b(0, 143, 255), "countertop"},
    {cv::Vec3b(51, 255, 0), "stove"},
    {cv::Vec3b(0, 255, 41), "kitchen island"},
    {cv::Vec3b(0, 255, 173), "computer"},
    {cv::Vec3b(0, 255, 194), "television"},
    {cv::Vec3b(0, 255, 10), "bottle"},
    {cv::Vec3b(184, 0, 255), "washer"},
    {cv::Vec3b(0, 214, 255), "stool"},
    {cv::Vec3b(70, 184, 160), "bag"},
    {cv::Vec3b(71, 255, 0), "oven"},
    {cv::Vec3b(255, 0, 235), "microwave"},
    {cv::Vec3b(214, 255, 0), "dishwasher"},
    {cv::Vec3b(0, 204, 255), "screen"},
    {cv::Vec3b(0, 255, 204), "vase"},
    {cv::Vec3b(0, 92, 255), "monitor"},
    {cv::Vec3b(184, 255, 0), "board"},
    {cv::Vec3b(102, 255, 0), "clock"},
    {cv::Vec3b(173, 0, 255), "trash"},

};


// Function to load colors from a .txt file
std::vector<cv::Vec3b> loadColors(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<cv::Vec3b> colors;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int b, g, r;
        ss >> b >> g >> r;
        colors.emplace_back(cv::Vec3b(b, g, r));
    }
    return colors;
}

class SemanticMaskToBoundingBox : public rclcpp::Node
{
public:
    SemanticMaskToBoundingBox()
    : Node("semantic_mask_to_bounding_box")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/sem_image", 10, std::bind(&SemanticMaskToBoundingBox::listener_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/bounding_boxes", 10);
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/image_with_bboxes", 10);
    }

    std::string filename = "/workspace/ws/src/classes/colors.txt";
    // Load the colors
    std::vector<cv::Vec3b> colors = loadColors(filename);
private:

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;

    void listener_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        vision_msgs::msg::Detection2DArray detection_array;
        cv::Mat image_with_bboxes = cv_ptr->image.clone();

        // Process each color in the mapping
        for (const auto& color_object_pair : color_to_object)
        {
            const cv::Vec3b& color = color_object_pair.first;
            const std::string& object_name = color_object_pair.second;

            // Create a binary mask for the current color
            cv::Mat mask;
            cv::inRange(cv_ptr->image, color, color, mask);

            // int width = mask.cols;
            // int height = mask.rows;
            // int channels = cv_ptr->image.channels();

            // std::cout << "Image width: " << width << std::endl;
            // std::cout << "Image height: " << height << std::endl;
            // std::cout << "Number of channels: " << channels << std::endl;

            // cv::imshow("Display window", mask);
            // cv::waitKey(0);

            // continue;
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours)
            {
                // std::cout<<"contour"<<std::endl;
                // std::cout<<object_name<<std::endl;
                cv::Rect bounding_rect = cv::boundingRect(contour);
                vision_msgs::msg::Detection2D detection;
                vision_msgs::msg::BoundingBox2D bbox;
                bbox.center.position.x = bounding_rect.x + bounding_rect.width / 2.0;
                bbox.center.position.y = bounding_rect.y + bounding_rect.height / 2.0;
                bbox.size_x = bounding_rect.width;
                bbox.size_y = bounding_rect.height;
                detection.bbox = bbox;
                detection.results.push_back(vision_msgs::msg::ObjectHypothesisWithPose());
                detection.id = object_name;
                detection_array.detections.push_back(detection);

                // Draw the bounding box on the image
                cv::rectangle(image_with_bboxes, bounding_rect, cv::Scalar(0, 255, 0), 2);
                cv::putText(image_with_bboxes, object_name, cv::Point(bounding_rect.x, bounding_rect.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
            }
        }
        detection_array.header = msg->header;
        publisher_->publish(detection_array);
        // std::cout<<"publihed"<<std::endl;

        // Convert the image with bounding boxes to a ROS message
        cv_bridge::CvImage out_msg;
        out_msg.header = msg->header; // Use the same timestamp and frame_id as the input image
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = image_with_bboxes;

        // Publish the image with bounding boxes
        image_publisher_->publish(*out_msg.toImageMsg());
    }


};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SemanticMaskToBoundingBox>());
    rclcpp::shutdown();
    return 0;
}
