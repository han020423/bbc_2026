#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from xycar_msgs.msg import XycarMotor


class TrackDriverNode(Node):
    def __init__(self):
        super().__init__("driver")
        self.get_logger().info("----- Xycar yellow-center sliding-window node started -----")

        self.image = None
        self.lidar_ranges = None
        self.motor_msg = XycarMotor()
        self.bridge = CvBridge()

        self.MAX_ANGLE = 100.0
        self.MAX_SPEED = 100.0
        self.LIDAR_MIN_VALID = 0.35

        # Yellow dashed centerline first, white solid-line fallback.
        self.KP = 0.58
        self.BASE_SPEED = 16.0
        self.MID_SPEED = 10.0
        self.CURVE_SPEED = 5.0
        self.LOST_SPEED = 0.0
        self.DEADBAND_PX = 10.0
        self.MAX_ANGLE_DELTA = 35.0

        self.ROI_TOP_RATIO = 0.55
        self.ROI_BOTTOM_RATIO = 0.99
        self.NWINDOWS = 12
        self.WINDOW_MARGIN = 75
        self.MINPIX = 10
        self.MIN_FIT_PIXELS = 45
        self.YELLOW_MIN_FIT_PIXELS = 45
        self.MIN_BOTTOM_HITS = 1
        self.LANE_WIDTH_PX = 285.0
        self.YELLOW_MIN_PEAK = 260
        self.WHITE_MIN_PEAK = 420

        self.yellow_fit = None
        self.white_fit = None
        self.center_fit = None
        self.last_angle = 0.0
        self.fail_count = 0
        self.frame_count = 0

        self.show_debug = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

        self.motor_pub = self.create_publisher(XycarMotor, "xycar_motor", 10)
        self.sub_front = self.create_subscription(
            Image, "/usb_cam/image_raw/front", self.cam_callback, qos_profile_sensor_data)
        self.sub_lidar = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, qos_profile_sensor_data)

        self.get_logger().info("Track Driver Node Initialized")

    def cam_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            self.get_logger().warn(f"camera conversion failed: {exc}")

    def lidar_callback(self, msg):
        self.lidar_ranges = list(msg.ranges)

    def drive(self, angle, speed):
        self.motor_msg.angle = float(np.clip(angle, -self.MAX_ANGLE, self.MAX_ANGLE))
        self.motor_msg.speed = float(np.clip(speed, -self.MAX_SPEED, self.MAX_SPEED))
        self.motor_pub.publish(self.motor_msg)

    def make_lane_masks(self, image):
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, np.array([15, 55, 70]), np.array([43, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 120, 255]))

        top_y = int(height * self.ROI_TOP_RATIO)
        bottom_y = int(height * self.ROI_BOTTOM_RATIO)
        roi_polygon = np.array([[
            (0, bottom_y),
            (int(width * 0.06), top_y),
            (int(width * 0.94), top_y),
            (width - 1, bottom_y),
        ]], dtype=np.int32)

        roi_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(roi_mask, roi_polygon, 255)
        yellow_mask = cv2.bitwise_and(yellow_mask, roi_mask)
        white_mask = cv2.bitwise_and(white_mask, roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        return yellow_mask, white_mask, roi_polygon, top_y, bottom_y

    def empty_lane(self):
        return {
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "windows": [],
            "bottom_hits": 0,
            "valid": False,
        }

    def find_lane_bases(self, lane_mask, min_peak, previous_fit=None):
        height, width = lane_mask.shape[:2]
        hist = np.sum(lane_mask[int(height * 0.72):, :], axis=0)
        midpoint = width // 2

        left_hist = hist[:midpoint]
        right_hist = hist[midpoint:]
        bases = []

        if np.max(left_hist) > min_peak:
            bases.append(int(np.argmax(left_hist)))
        if np.max(right_hist) > min_peak:
            bases.append(int(midpoint + np.argmax(right_hist)))

        if not bases and previous_fit is not None:
            y_bottom = height - 1
            bases.append(int(np.clip(np.polyval(previous_fit, y_bottom), 0, width - 1)))

        return bases

    def track_one_lane(self, lane_mask, base_x, top_y, bottom_y):
        height, width = lane_mask.shape[:2]
        nonzero_y, nonzero_x = lane_mask.nonzero()
        window_height = max(1, (bottom_y - top_y) // self.NWINDOWS)

        if base_x is None:
            return self.empty_lane()

        current_x = int(base_x)
        lane_inds = []
        windows = []
        bottom_hits = 0

        for window in range(self.NWINDOWS):
            win_y_high = bottom_y - window * window_height
            win_y_low = max(top_y, win_y_high - window_height)
            if win_y_high <= top_y:
                break

            win_x_low = int(np.clip(current_x - self.WINDOW_MARGIN, 0, width - 1))
            win_x_high = int(np.clip(current_x + self.WINDOW_MARGIN, 0, width - 1))
            good_inds = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_low) & (nonzero_x < win_x_high)
            ).nonzero()[0]

            hit = len(good_inds) > self.MINPIX
            if hit:
                lane_inds.append(good_inds)
                current_x = int(np.mean(nonzero_x[good_inds]))
                if window <= 2:
                    bottom_hits += 1

            windows.append((win_x_low, win_y_low, win_x_high, win_y_high, hit))

        lane_inds = np.concatenate(lane_inds) if lane_inds else np.array([], dtype=np.int64)
        return {
            "x": nonzero_x[lane_inds],
            "y": nonzero_y[lane_inds],
            "windows": windows,
            "bottom_hits": bottom_hits,
            "valid": False,
        }

    def fit_lane(self, lane, min_pixels=None, require_bottom=True):
        if min_pixels is None:
            min_pixels = self.MIN_FIT_PIXELS
        if len(lane["x"]) < min_pixels:
            return None
        if require_bottom and lane["bottom_hits"] < self.MIN_BOTTOM_HITS:
            return None
        if not require_bottom:
            y_span = float(np.max(lane["y"]) - np.min(lane["y"])) if len(lane["y"]) else 0.0
            if y_span < 38.0:
                return None
        elif lane["bottom_hits"] < self.MIN_BOTTOM_HITS:
            return None
        degree = 2 if len(lane["x"]) >= 150 else 1
        fit = np.polyfit(lane["y"], lane["x"], degree)
        if degree == 1:
            fit = np.array([0.0, fit[0], fit[1]], dtype=np.float64)
        lane["valid"] = True
        return fit

    def choose_white_lane(self, lanes, fits, width, height):
        candidates = []
        y_bottom = height - 1
        for lane, fit in zip(lanes, fits):
            if fit is None:
                continue
            x_bottom = float(np.polyval(fit, y_bottom))
            if not (-width * 0.15 <= x_bottom <= width * 1.15):
                continue
            score = len(lane["x"]) + 250 * lane["bottom_hits"] - 0.12 * abs(x_bottom - width / 2.0)
            candidates.append((score, lane, fit, x_bottom))

        if not candidates:
            return None, None, "lost"

        _, lane, fit, x_bottom = max(candidates, key=lambda item: item[0])
        side = "left_white" if x_bottom < width / 2.0 else "right_white"
        return lane, fit, side

    def find_best_lane(self, lane_mask, min_peak, previous_fit, top_y, bottom_y, width, height,
                       min_pixels=None, require_bottom=True):
        bases = self.find_lane_bases(lane_mask, min_peak, previous_fit)
        lanes = [self.track_one_lane(lane_mask, base, top_y, bottom_y) for base in bases]
        fits = [self.fit_lane(lane, min_pixels=min_pixels, require_bottom=require_bottom) for lane in lanes]

        candidates = []
        for lane, fit in zip(lanes, fits):
            if fit is None:
                continue
            if require_bottom:
                y_eval = height - 1
            else:
                y_eval = int(np.percentile(lane["y"], 80))
            x_eval = float(np.polyval(fit, y_eval))
            if not (-width * 0.15 <= x_eval <= width * 1.15):
                continue
            bottom_weight = 250 if require_bottom else 80
            score = len(lane["x"]) + bottom_weight * lane["bottom_hits"] - 0.08 * abs(x_eval - width / 2.0)
            candidates.append((score, lane, fit, x_eval))

        if not candidates:
            return None, None, None, lanes, fits

        _, lane, fit, x_eval = max(candidates, key=lambda item: item[0])
        side = "left" if x_eval < width / 2.0 else "right"
        return lane, fit, side, lanes, fits

    def make_center_from_white(self, white_fit, side):
        center_fit = white_fit.copy()
        if side == "left_white":
            center_fit[2] += self.LANE_WIDTH_PX * 0.50
        else:
            center_fit[2] -= self.LANE_WIDTH_PX * 0.50
        return center_fit

    def is_curve_from_fit(self, fit, width, height):
        y_target = int(height * 0.64)
        y_bottom = height - 1
        target_x = float(np.polyval(fit, y_target))
        bottom_x = float(np.polyval(fit, y_bottom))
        error = target_x - width / 2.0
        slope = 2.0 * fit[0] * y_target + fit[1]
        heading = float(np.degrees(np.arctan(slope)))
        return abs(error) > 80.0 or abs(heading) > 12.0 or abs(self.last_angle) > 25.0

    def is_sharp_curve_from_fit(self, fit, width, height):
        y_target = int(height * 0.64)
        target_x = float(np.polyval(fit, y_target))
        error = target_x - width / 2.0
        slope = 2.0 * fit[0] * y_target + fit[1]
        heading = float(np.degrees(np.arctan(slope)))
        return abs(error) > 150.0 or abs(heading) > 24.0 or abs(self.last_angle) > 55.0

    def blend_center_with_white(self, yellow_fit, white_fit, white_side):
        white_center = self.make_center_from_white(white_fit, f"{white_side}_white")
        # In curves, white solid line should dominate because dashed yellow can jump.
        return 0.35 * yellow_fit + 0.65 * white_center

    def calculate_error(self, center_fit, width, height):
        y_target = int(height * 0.64)
        target_x = float(np.polyval(center_fit, y_target))
        target_x = float(np.clip(target_x, -width * 0.15, width * 1.15))
        error = float(np.clip(target_x - width / 2.0, -180.0, 180.0))
        return error, target_x, y_target

    def p_control(self, error):
        if abs(error) < self.DEADBAND_PX:
            error = 0.0
        raw_angle = float(np.clip(self.KP * error, -self.MAX_ANGLE, self.MAX_ANGLE))
        delta = float(np.clip(raw_angle - self.last_angle, -self.MAX_ANGLE_DELTA, self.MAX_ANGLE_DELTA))
        angle = float(np.clip(self.last_angle + delta, -self.MAX_ANGLE, self.MAX_ANGLE))
        self.last_angle = angle
        return angle

    def select_speed(self, angle, mode):
        if mode == "lost":
            return self.LOST_SPEED
        abs_angle = abs(angle)
        if abs_angle > 45.0:
            return self.CURVE_SPEED
        if abs_angle > 18.0:
            return self.MID_SPEED
        return self.BASE_SPEED

    def decide_control(self, image):
        height, width = image.shape[:2]
        yellow_mask, white_mask, roi_polygon, top_y, bottom_y = self.make_lane_masks(image)

        yellow_lane, yellow_fit, yellow_side, yellow_lanes, yellow_fits = self.find_best_lane(
            yellow_mask, self.YELLOW_MIN_PEAK, self.yellow_fit, top_y, bottom_y, width, height,
            min_pixels=self.YELLOW_MIN_FIT_PIXELS, require_bottom=False)
        white_lane, white_fit, white_side, white_lanes, white_fits = self.find_best_lane(
            white_mask, self.WHITE_MIN_PEAK, self.white_fit, top_y, bottom_y, width, height,
            min_pixels=self.MIN_FIT_PIXELS, require_bottom=False)

        debug = {
            "yellow_mask": yellow_mask,
            "white_mask": white_mask,
            "roi_polygon": roi_polygon,
            "yellow_lanes": yellow_lanes,
            "white_lanes": white_lanes,
            "yellow_fits": yellow_fits,
            "white_fits": white_fits,
            "yellow_fit": yellow_fit,
            "white_fit": white_fit,
            "center_fit": None,
            "mode": "lost",
        }

        if yellow_fit is not None:
            self.fail_count = 0
            self.yellow_fit = yellow_fit
            self.white_fit = white_fit
            if white_fit is not None and self.is_sharp_curve_from_fit(yellow_fit, width, height):
                self.center_fit = self.make_center_from_white(white_fit, f"{white_side}_white")
                mode = "white_curve"
            elif white_fit is not None and self.is_curve_from_fit(yellow_fit, width, height):
                self.center_fit = self.blend_center_with_white(yellow_fit, white_fit, white_side)
                mode = "yellow_white_curve"
            else:
                self.center_fit = yellow_fit.copy()
                mode = "yellow_center"
        elif white_fit is not None:
            self.fail_count = 0
            self.white_fit = white_fit
            self.center_fit = self.make_center_from_white(white_fit, f"{white_side}_white")
            mode = f"{white_side}_white"
        else:
            self.fail_count += 1
            self.yellow_fit = None
            self.white_fit = None
            self.center_fit = None
            self.last_angle = 0.0
            return 0.0, self.LOST_SPEED, debug

        error, target_x, y_target = self.calculate_error(self.center_fit, width, height)
        angle = self.p_control(error)
        speed = self.select_speed(angle, mode)

        debug.update({
            "center_fit": self.center_fit,
            "center_target": target_x,
            "y_target": y_target,
            "error": error,
            "mode": mode,
        })
        return angle, speed, debug

    def front_obstacle_distance(self):
        if not self.lidar_ranges:
            return math.inf

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        if not np.any(np.isfinite(ranges)):
            return math.inf

        center = len(ranges) // 2
        half_width = max(8, len(ranges) // 36)
        indices = np.arange(center - half_width, center + half_width + 1) % len(ranges)
        front_values = ranges[indices]
        front_values = front_values[np.isfinite(front_values)]
        front_values = front_values[front_values > self.LIDAR_MIN_VALID]
        if len(front_values) == 0:
            return math.inf
        return float(np.percentile(front_values, 20))

    def draw_fit(self, view, fit, color, thickness=3):
        if fit is None:
            return
        height, width = view.shape[:2]
        ys = np.linspace(int(height * self.ROI_TOP_RATIO), height - 1, 50)
        xs = fit[0] * ys ** 2 + fit[1] * ys + fit[2]
        pts = []
        for x, y in zip(xs, ys):
            if -width * 0.20 <= x < width * 1.20:
                pts.append((int(np.clip(x, 0, width - 1)), int(y)))
        if len(pts) >= 2:
            cv2.polylines(view, [np.array(pts, dtype=np.int32)], False, color, thickness)

    def make_window_debug(self, yellow_mask, white_mask, yellow_lanes, white_lanes):
        combined = cv2.bitwise_or(yellow_mask, white_mask)
        debug = np.dstack((combined, combined, combined))
        for lane in yellow_lanes:
            for x1, y1, x2, y2, hit in lane["windows"]:
                color = (0, 255, 255) if hit and lane["valid"] else (0, 90, 180)
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        for lane in white_lanes:
            for x1, y1, x2, y2, hit in lane["windows"]:
                color = (0, 255, 0) if hit and lane["valid"] else (0, 0, 255)
                cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        return debug

    def show_debug_image(self, image, debug, angle, speed, front_dist):
        if not self.show_debug:
            return

        view = image.copy()
        height, width = view.shape[:2]
        cv2.polylines(view, debug["roi_polygon"], True, (255, 255, 0), 2)
        for fit in debug.get("yellow_fits", []):
            self.draw_fit(view, fit, (0, 150, 150), 1)
        for fit in debug.get("white_fits", []):
            self.draw_fit(view, fit, (180, 180, 180), 1)
        self.draw_fit(view, debug.get("yellow_fit"), (0, 255, 255), 3)
        self.draw_fit(view, debug.get("white_fit"), (255, 255, 255), 3)
        self.draw_fit(view, debug.get("center_fit"), (0, 0, 255), 4)

        if "center_target" in debug:
            target_x = int(np.clip(debug["center_target"], 0, width - 1))
            y_target = int(debug.get("y_target", height * 0.64))
            cv2.line(view, (width // 2, height - 1), (target_x, y_target), (0, 0, 255), 3)
            cv2.circle(view, (target_x, y_target), 7, (0, 255, 0), -1)

        cv2.putText(
            view,
            f"angle={angle:.1f} speed={speed:.1f} front={front_dist:.2f} fail={self.fail_count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            view,
            f"mode={debug.get('mode')} err={debug.get('error', 0):.1f}",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.imshow("track_drive camera debug", view)
        cv2.imshow("yellow white sliding windows", self.make_window_debug(
            debug["yellow_mask"], debug["white_mask"], debug["yellow_lanes"], debug["white_lanes"]))
        cv2.waitKey(1)

    def main_loop(self):
        self.get_logger().info("======================================")
        self.get_logger().info("  Y E L L O W   C E N T E R           ")
        self.get_logger().info("======================================")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.03)

            if self.image is None:
                self.drive(angle=0.0, speed=0.0)
                time.sleep(0.03)
                continue

            angle, speed, debug = self.decide_control(self.image)
            front_dist = self.front_obstacle_distance()

            if front_dist < 0.8:
                angle, speed = 0.0, 0.0
            elif front_dist < 1.5:
                speed = min(speed, 8.0)

            self.drive(angle=angle, speed=speed)

            self.frame_count += 1
            if self.frame_count % 2 == 0:
                self.show_debug_image(self.image, debug, angle, speed, front_dist)

            time.sleep(0.02)


def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.drive(angle=0.0, speed=0.0)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
