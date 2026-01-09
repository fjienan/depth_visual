import threading
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import logging


class Test_UI(Node):
    def __init__(self, topic_name='/k4a/depth_to_rgb/image_raw',image_type = 'depth'):
        super().__init__('Test_UI')
        self.bridge = CvBridge() # ros2 opencv bridge
        self.subscriber = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10
        )
        self.image = None
        # timestamp of last received image
        self.last_msg_time = 0.0
        # simple counter for received msgs
        self.msg_count = 0
        # lock to protect access to self.image between threads
        self._image_lock = threading.Lock()

        # desire encoding based on image type
        if image_type == 'color':
            self.encoding = 'bgr8'
        elif image_type == 'depth':
            self.encoding = '32FC1'
        else:
            self.get_logger().error(f"Unsupported image type: {image_type}. Defaulting to 'bgr8'.")
            self.encoding = 'bgr8'

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.encoding)
            print("dtype:", cv_image.dtype, "shape:", cv_image.shape, "min:", cv_image.min(), "max:", cv_image.max())
            self.msg_count += 1
            self.last_msg_time = time.time()
            # log first few messages to confirm reception
            if self.msg_count <= 5 or (self.msg_count % 100) == 0:
                self.get_logger().info(f"Received image #{self.msg_count}")
            # store latest frame for main thread to display (do not call GUI from callback)
            with self._image_lock:
                self.image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def visualize(self, cv_image):
        if cv_image is not None:
            cv2.imshow('Image Visualization', cv_image)
            cv2.waitKey(1)
            
def main(args=None):
    # Configure rudimentary logging to the console
    logging.basicConfig(level=logging.INFO)

    rclpy.init(args=args)
    test_ui = Test_UI()

    # Start spinning in a background thread so the main thread can
    # handle OpenCV GUI (imshow + waitKey) which on many platforms
    # must run in the main thread.
    spin_thread = threading.Thread(target=rclpy.spin, args=(test_ui,), daemon=True)
    spin_thread.start()

    try:
        # main loop handles GUI event processing and checks for incoming images
        while rclpy.ok():
            # get latest frame from callback thread
            img = None
            with test_ui._image_lock:
                if test_ui.image is not None:
                    img = test_ui.image.copy()
                    # optionally clear stored image to reduce redraws
                    # test_ui.image = None

            if img is not None:
                # show image in main thread (Qt-safe)
                test_ui.visualize(img)

            # warn if we haven't received any images for 5 seconds
            if test_ui.msg_count == 0 and time.time() - test_ui.last_msg_time > 5:
                # only print once per loop; use get_logger so it's visible with ROS logs
                test_ui.get_logger().error('No images received yet. Check topic name and that camera node is publishing.')
                # sleep a bit to avoid spamming
                time.sleep(1.0)
            else:
                # small sleep to yield CPU
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        # clean shutdown
        try:
            test_ui.get_logger().info('Shutting down test_ui')
        except Exception:
            pass
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        try:
            test_ui.destroy_node()
        except Exception:
            pass
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main()