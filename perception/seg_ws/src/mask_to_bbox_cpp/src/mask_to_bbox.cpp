#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <map>

// Custom comparator for cv::Vec3b
struct Vec3bCompare {
    bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const {
        return std::tie(lhs[0], lhs[1], lhs[2]) < std::tie(rhs[0], rhs[1], rhs[2]);
    }
};

// Function to load colors from a .txt file
std::vector<cv::Vec3b> loadColors(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<cv::Vec3b> colors;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        int b, g, r;
        ss >> r >> g >> b;
        colors.emplace_back(cv::Vec3b(b, g, r));
    }
    return colors;
}

// std::vector<std::string> loadSentences(const std::string& filename) {
//     std::ifstream file(filename);
//     std::vector<std::string> sentences;
//     std::string line;

//     while (std::getline(file, line)) {
//         // Store the first 10 characters of the sentence
//         if (line.length() > 10) {
//             sentences.push_back(line.substr(0, 10));
//         } else {
//             sentences.push_back(line);
//         }
//     }
//     return sentences;
// }


std::vector<std::string> loadSentences(const std::string& filename) {
  std::ifstream file(filename);
  std::vector<std::string> sentences;
  std::string line;

  while (std::getline(file, line)) {
    // Find the first comma
    size_t comma_index = line.find(",");

    // If there's a comma, store the substring up to the comma
    if (comma_index != std::string::npos) {
      sentences.push_back(line.substr(0, comma_index));
    } else {
      // If there's no comma, store the entire line
      sentences.push_back(line);
    }
  }
  return sentences;
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

    // std::string colors_filename = "/workspace/ws/src/classes/colors.txt";
    std::string colors_filename = "/workspace/ws/src/classes/colors_new.txt";
    // Load the colors
    std::vector<cv::Vec3b> all_object_colors = loadColors(colors_filename);

    // std::string class_name_filename = "/workspace/ws/src/classes/model_classes.txt";
    std::string class_name_filename = "/workspace/ws/src/classes/model_classes_new.txt";

    // Load the sentences
    std::vector<std::string> class_names = loadSentences(class_name_filename);

    // std::set<int> skip_indices = {0, 1131, 150, 1782, 1783, 1784, 1785};
    std::set<int> skip_indices = {0, 71, 132, 265, 493, 557};

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
        int ii=0;
        for (const auto& object_color : all_object_colors)
        {
            // Check if ii belongs to the skip list
            if (skip_indices.find(ii) != skip_indices.end()) {
                ii += 1;
                continue; // Skip this iteration
            }
            const cv::Vec3b& color = object_color;
            // std::cout<<color<<std::endl;
            const std::string& object_name = class_names[ii];
            ii+=1;

            // Create a binary mask for the current color
            cv::Mat mask;
            cv::inRange(cv_ptr->image, color, color, mask);
            if (cv::countNonZero(mask) == 0) {

                continue; // Skip to the next iteration if no points are found
            }

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
