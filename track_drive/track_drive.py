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
        self.YELLOW_CENTER_KP = 0.25
        self.YELLOW_D_GAIN = 0.55
        self.YELLOW_D_FILTER_ALPHA = 0.18
        self.YELLOW_D_MAX = 18.0
        self.BASE_SPEED = 22.0
        self.MID_SPEED = 14.0
        self.CURVE_SPEED = 6.0
        self.SCHOOL_ZONE_SPEED = 6.0
        self.SPEED_ACCEL_FROM_SHARP = 0.25
        self.SPEED_ACCEL_FROM_MID = 0.60
        self.SPEED_ACCEL_NORMAL = 0.80
        self.LOST_SPEED = 0.0
        self.DEADBAND_PX = 10.0
        self.MAX_ANGLE_DELTA = 35.0
        self.CURVE_HOLD_FRAMES = 12
        self.SHARP_CURVE_HOLD_FRAMES = 6
        self.CURVE_HEADING_DEG = 18.0
        self.CURVE_QUADRATIC = 0.0015
        self.SHARP_CURVE_HEADING_DEG = 30.0
        self.SHARP_CURVE_QUADRATIC = 0.0040
        self.WHITE_CURVE_ENTER_FRAMES = 4
        self.WHITE_CURVE_EXIT_FRAMES = 6
        self.WHITE_CURVE_MIN_HEADING_DEG = 30.0
        self.WHITE_CURVE_COMBINED_HEADING_DEG = 22.0
        self.WHITE_CURVE_COMBINED_QUADRATIC = 0.0020
        self.WHITE_CURVE_WHITE_JUMP_PX = 40.0
        self.SCHOOL_OUTER_BAND_RATIO = 0.28
        self.SCHOOL_OUTER_MIN_PIXELS = 120
        self.SCHOOL_OUTER_MIN_YSPAN = 60.0

        self.ROI_TOP_RATIO = 0.55
        self.ROI_BOTTOM_RATIO = 0.99
        self.NWINDOWS = 12
        self.WINDOW_MARGIN = 75
        self.MINPIX = 10
        self.MIN_FIT_PIXELS = 45
        self.YELLOW_MIN_FIT_PIXELS = 45
        self.MIN_BOTTOM_HITS = 1
        self.LANE_WIDTH_PX = 400.0
        self.YELLOW_MIN_PEAK = 260
        self.WHITE_MIN_PEAK = 420

        self.yellow_fit = None
        self.white_fit = None
        self.center_fit = None
        self.last_angle = 0.0
        self.last_command_speed = 0.0
        self.prev_yellow_control_error = None
        self.filtered_yellow_error_delta = 0.0
        self.curve_hold_remaining = 0
        self.sharp_curve_hold_remaining = 0
        self.white_curve_active = False
        self.sharp_curve_confirm_count = 0
        self.straight_confirm_count = 0
        self.sharp_curve_direction = 0
        self.prev_white_curve_candidate_x = None
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

        # Simulator lane colors: keep only vivid yellow and near-neutral white.
        yellow_mask = cv2.inRange(hsv, np.array([18, 125, 150]), np.array([40, 255, 255]))
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))

        # A lane marking must touch the gray road surface. This removes white
        # trees and other bright objects surrounded by green scenery.
        road_mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 40, 150]))
        road_support = cv2.dilate(road_mask, np.ones((17, 17), np.uint8), iterations=1)
        yellow_mask = cv2.bitwise_and(yellow_mask, road_support)
        white_mask = cv2.bitwise_and(white_mask, road_support)

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

    def detect_school_zone_outer_yellow(self, yellow_mask, top_y, bottom_y):
        """Detect the two long outer yellow boundaries of a school zone."""
        height, width = yellow_mask.shape[:2]
        band = int(width * self.SCHOOL_OUTER_BAND_RATIO)

        def support(region):
            ys, _ = region.nonzero()
            if len(ys) == 0:
                return 0, 0.0
            return len(ys), float(np.max(ys) - np.min(ys))

        left_region = yellow_mask[top_y:bottom_y, :band]
        right_region = yellow_mask[top_y:bottom_y, width - band:width]
        left_count, left_span = support(left_region)
        right_count, right_span = support(right_region)
        detected = (
            left_count >= self.SCHOOL_OUTER_MIN_PIXELS and
            right_count >= self.SCHOOL_OUTER_MIN_PIXELS and
            left_span >= self.SCHOOL_OUTER_MIN_YSPAN and
            right_span >= self.SCHOOL_OUTER_MIN_YSPAN
        )
        return detected, left_count, right_count

    def find_school_center_yellow(self, yellow_mask, top_y, bottom_y, width, height):
        """Track the dashed yellow marking seeded at the image center."""
        lane = self.track_one_lane(yellow_mask, width // 2, top_y, bottom_y)
        fit = self.fit_lane(
            lane,
            min_pixels=self.YELLOW_MIN_FIT_PIXELS,
            require_bottom=False,
        )
        if fit is None:
            return None, None, None
        y_reference = int(height * 0.64)
        x_reference = float(np.polyval(fit, y_reference))
        if abs(x_reference - width / 2.0) > width * 0.22:
            return None, None, None
        side = "left" if x_reference < width / 2.0 else "right"
        return lane, fit, side

    def select_center_yellow(self, lanes, fits, width, height):
        """Pick the valid yellow marking closest to the image center."""
        y_reference = int(height * 0.64)
        candidates = []
        for lane, fit in zip(lanes, fits):
            if fit is None:
                continue
            x_reference = float(np.polyval(fit, y_reference))
            if -width * 0.15 <= x_reference <= width * 1.15:
                candidates.append((abs(x_reference - width / 2.0), lane, fit, x_reference))
        if not candidates:
            return None, None, None
        _, lane, fit, x_reference = min(candidates, key=lambda item: item[0])
        side = "left" if x_reference < width / 2.0 else "right"
        return lane, fit, side

    def make_center_from_white(self, white_fit, side):
        center_fit = white_fit.copy()
        if side == "left_white":
            center_fit[2] += self.LANE_WIDTH_PX * 0.50
        else:
            center_fit[2] -= self.LANE_WIDTH_PX * 0.50
        return center_fit

    def make_virtual_opposite_fit(self, yellow_fit, yellow_side, white_fit, white_side):
        # Debug only: show the opposite lane boundary when exactly one side is visible.
        if yellow_fit is not None and white_fit is None:
            virtual_fit = yellow_fit.copy()
            shift = self.LANE_WIDTH_PX if yellow_side == "left" else -self.LANE_WIDTH_PX
            virtual_fit[2] += shift
            return virtual_fit, "virtual_white"

        if white_fit is not None and yellow_fit is None:
            virtual_fit = white_fit.copy()
            shift = self.LANE_WIDTH_PX if white_side == "left" else -self.LANE_WIDTH_PX
            virtual_fit[2] += shift
            return virtual_fit, "virtual_yellow"

        return None, ""

    def curve_metrics(self, fit, height):
        if fit is None:
            return 0.0, 0.0
        y_target = int(height * 0.64)
        slope = 2.0 * fit[0] * y_target + fit[1]
        heading = float(np.degrees(np.arctan(slope)))
        return heading, abs(float(fit[0]))

    def curve_geometry_label(self, heading, quadratic):
        if (abs(heading) > self.SHARP_CURVE_HEADING_DEG or
                quadratic > self.SHARP_CURVE_QUADRATIC):
            return "sharp"
        if abs(heading) > self.CURVE_HEADING_DEG or quadratic > self.CURVE_QUADRATIC:
            return "curve"
        return "straight"

    def update_white_curve_mode(self, yellow_fit, white_fit, height):
        """Enter white mode only after persistent sharp geometry and white continuity."""
        if yellow_fit is None or white_fit is None:
            self.white_curve_active = False
            self.sharp_curve_confirm_count = 0
            self.straight_confirm_count = 0
            self.sharp_curve_direction = 0
            self.prev_white_curve_candidate_x = None
            return False

        heading, quadratic = self.curve_metrics(yellow_fit, height)
        raw_sharp = (
            abs(heading) > self.WHITE_CURVE_MIN_HEADING_DEG or
            (abs(heading) > self.WHITE_CURVE_COMBINED_HEADING_DEG and
             quadratic > self.WHITE_CURVE_COMBINED_QUADRATIC)
        )
        raw_straight = (
            abs(heading) <= self.CURVE_HEADING_DEG and
            quadratic <= self.CURVE_QUADRATIC
        )
        direction = 1 if heading >= 0.0 else -1
        y_reference = int(height * 0.64)
        white_x = float(np.polyval(white_fit, y_reference))

        if not self.white_curve_active:
            white_continuous = (
                self.prev_white_curve_candidate_x is None or
                abs(white_x - self.prev_white_curve_candidate_x) <= self.WHITE_CURVE_WHITE_JUMP_PX
            )
            if raw_sharp and white_continuous:
                if direction == self.sharp_curve_direction:
                    self.sharp_curve_confirm_count += 1
                else:
                    self.sharp_curve_direction = direction
                    self.sharp_curve_confirm_count = 1
                self.prev_white_curve_candidate_x = white_x
                if self.sharp_curve_confirm_count >= self.WHITE_CURVE_ENTER_FRAMES:
                    self.white_curve_active = True
                    self.straight_confirm_count = 0
            else:
                self.sharp_curve_confirm_count = 0
                self.sharp_curve_direction = 0
                self.prev_white_curve_candidate_x = white_x if raw_sharp else None
            return self.white_curve_active

        if raw_straight:
            self.straight_confirm_count += 1
            if self.straight_confirm_count >= self.WHITE_CURVE_EXIT_FRAMES:
                self.white_curve_active = False
                self.sharp_curve_confirm_count = 0
                self.sharp_curve_direction = 0
                self.prev_white_curve_candidate_x = None
        else:
            self.straight_confirm_count = 0
        return self.white_curve_active

    def is_curve_from_fit(self, fit, width, height):
        y_target = int(height * 0.64)
        slope = 2.0 * fit[0] * y_target + fit[1]
        heading = float(np.degrees(np.arctan(slope)))
        curvature = abs(float(fit[0]))
        geometric_curve = (
            abs(heading) > self.CURVE_HEADING_DEG or
            curvature > self.CURVE_QUADRATIC
        )

        if geometric_curve:
            self.curve_hold_remaining = self.CURVE_HOLD_FRAMES
            return True
        if self.curve_hold_remaining > 0:
            self.curve_hold_remaining -= 1
            return True
        return False

    def is_sharp_curve_from_fit(self, fit, width, height):
        y_target = int(height * 0.64)
        slope = 2.0 * fit[0] * y_target + fit[1]
        heading = float(np.degrees(np.arctan(slope)))
        curvature = abs(float(fit[0]))
        geometric_sharp_curve = (
            abs(heading) > self.SHARP_CURVE_HEADING_DEG or
            curvature > self.SHARP_CURVE_QUADRATIC
        )

        if geometric_sharp_curve:
            self.sharp_curve_hold_remaining = self.SHARP_CURVE_HOLD_FRAMES
            self.curve_hold_remaining = self.CURVE_HOLD_FRAMES
            return True
        if self.sharp_curve_hold_remaining > 0:
            self.sharp_curve_hold_remaining -= 1
            return True
        return False

    def calculate_error(self, center_fit, width, height):
        y_target = int(height * 0.64)
        target_x = float(np.polyval(center_fit, y_target))
        target_x = float(np.clip(target_x, -width * 0.15, width * 1.15))
        error = float(np.clip(target_x - width / 2.0, -180.0, 180.0))
        return error, target_x, y_target

    def p_control(self, error, mode):
        if abs(error) < self.DEADBAND_PX:
            error = 0.0

        d_term = 0.0
        if mode in ("yellow_center", "school_zone_yellow"):
            if self.prev_yellow_control_error is None:
                raw_error_delta = 0.0
            else:
                raw_error_delta = error - self.prev_yellow_control_error
            alpha = self.YELLOW_D_FILTER_ALPHA
            self.filtered_yellow_error_delta = (
                (1.0 - alpha) * self.filtered_yellow_error_delta +
                alpha * raw_error_delta
            )
            self.prev_yellow_control_error = error
            d_term = float(np.clip(
                self.YELLOW_D_GAIN * self.filtered_yellow_error_delta,
                -self.YELLOW_D_MAX,
                self.YELLOW_D_MAX,
            ))
        else:
            self.prev_yellow_control_error = None
            self.filtered_yellow_error_delta = 0.0

        p_gain = (
            self.YELLOW_CENTER_KP
            if mode in ("yellow_center", "school_zone_yellow")
            else self.KP
        )
        raw_angle = float(np.clip(
            p_gain * error + d_term, -self.MAX_ANGLE, self.MAX_ANGLE))
        delta = float(np.clip(raw_angle - self.last_angle, -self.MAX_ANGLE_DELTA, self.MAX_ANGLE_DELTA))
        angle = float(np.clip(self.last_angle + delta, -self.MAX_ANGLE, self.MAX_ANGLE))
        self.last_angle = angle
        return angle, d_term

    def select_speed(self, angle, mode):
        if mode == "lost":
            return self.LOST_SPEED
        if mode == "school_zone_yellow":
            return self.SCHOOL_ZONE_SPEED
        abs_angle = abs(angle)
        if abs_angle > 70.0:
            return self.CURVE_SPEED
        if abs_angle > 25.0:
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

        valid_yellow_count = sum(fit is not None for fit in yellow_fits)
        outer_yellow_detected, school_left_pixels, school_right_pixels = (
            self.detect_school_zone_outer_yellow(yellow_mask, top_y, bottom_y)
        )
        school_zone_yellow = False
        if outer_yellow_detected:
            school_lane, school_fit, school_side = self.find_school_center_yellow(
                yellow_mask, top_y, bottom_y, width, height)
            if school_fit is not None:
                yellow_lane, yellow_fit, yellow_side = school_lane, school_fit, school_side
                yellow_lanes.append(school_lane)
                yellow_fits.append(school_fit)
                school_zone_yellow = True

        curve_heading, curve_quadratic = self.curve_metrics(yellow_fit, height)
        curve_geometry = self.curve_geometry_label(curve_heading, curve_quadratic)

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
            "virtual_fit": None,
            "virtual_kind": "",
            "curve_heading": curve_heading,
            "curve_quadratic": curve_quadratic,
            "curve_geometry": curve_geometry,
            "valid_yellow_count": valid_yellow_count,
            "school_zone_detected": school_zone_yellow,
            "school_left_pixels": school_left_pixels,
            "school_right_pixels": school_right_pixels,
            "mode": "lost",
        }

        if yellow_fit is None or white_fit is None:
            self.curve_hold_remaining = 0
            self.sharp_curve_hold_remaining = 0

        if yellow_fit is not None:
            self.fail_count = 0
            self.yellow_fit = yellow_fit
            self.white_fit = white_fit
            if school_zone_yellow:
                self.white_curve_active = False
                self.sharp_curve_confirm_count = 0
                self.straight_confirm_count = 0
                self.sharp_curve_direction = 0
                self.prev_white_curve_candidate_x = None
                self.center_fit = yellow_fit.copy()
                mode = "school_zone_yellow"
            elif self.update_white_curve_mode(yellow_fit, white_fit, height):
                self.center_fit = self.make_center_from_white(white_fit, f"{white_side}_white")
                mode = "white_curve"
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
            self.prev_yellow_control_error = None
            self.filtered_yellow_error_delta = 0.0
            self.curve_hold_remaining = 0
            self.sharp_curve_hold_remaining = 0
            self.white_curve_active = False
            self.sharp_curve_confirm_count = 0
            self.straight_confirm_count = 0
            self.sharp_curve_direction = 0
            self.prev_white_curve_candidate_x = None
            return 0.0, self.LOST_SPEED, debug

        virtual_fit, virtual_kind = self.make_virtual_opposite_fit(
            yellow_fit, yellow_side, white_fit, white_side)

        error, target_x, y_target = self.calculate_error(self.center_fit, width, height)
        angle, d_term = self.p_control(error, mode)
        speed = self.select_speed(angle, mode)

        debug.update({
            "center_fit": self.center_fit,
            "virtual_fit": virtual_fit,
            "virtual_kind": virtual_kind,
            "center_target": target_x,
            "y_target": y_target,
            "error": error,
            "d_term": d_term,
            "curve_heading": curve_heading,
            "curve_quadratic": curve_quadratic,
            "curve_geometry": curve_geometry,
            "valid_yellow_count": valid_yellow_count,
            "school_zone_detected": school_zone_yellow,
            "school_left_pixels": school_left_pixels,
            "school_right_pixels": school_right_pixels,
            "white_curve_active": self.white_curve_active,
            "sharp_confirm": self.sharp_curve_confirm_count,
            "straight_confirm": self.straight_confirm_count,
            "mode": mode,
        })
        return angle, speed, debug

    def apply_speed_recovery_ramp(self, requested_speed):
        """Apply every deceleration immediately and recover each speed tier gradually."""
        if requested_speed <= 0.0:
            self.last_command_speed = 0.0
            return 0.0

        # Do not delay initial launch. Recovery limiting starts only after the
        # vehicle has already been commanded to move.
        if self.last_command_speed <= 0.0:
            self.last_command_speed = requested_speed
            return requested_speed

        if requested_speed < self.last_command_speed:
            self.last_command_speed = requested_speed
            return requested_speed

        if requested_speed > self.last_command_speed:
            if self.last_command_speed <= self.CURVE_SPEED:
                acceleration = self.SPEED_ACCEL_FROM_SHARP
            elif self.last_command_speed <= self.MID_SPEED:
                acceleration = self.SPEED_ACCEL_FROM_MID
            else:
                acceleration = self.SPEED_ACCEL_NORMAL
            requested_speed = min(requested_speed, self.last_command_speed + acceleration)

        self.last_command_speed = requested_speed
        return requested_speed

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

    def draw_dashed_fit(self, view, fit, color, thickness=2):
        if fit is None:
            return
        height, width = view.shape[:2]
        ys = np.linspace(int(height * self.ROI_TOP_RATIO), height - 1, 50)
        xs = fit[0] * ys ** 2 + fit[1] * ys + fit[2]
        pts = [
            (int(np.clip(x, 0, width - 1)), int(y))
            for x, y in zip(xs, ys)
            if -width * 0.20 <= x < width * 1.20
        ]
        for index in range(0, len(pts) - 1, 6):
            end = min(index + 3, len(pts) - 1)
            cv2.line(view, pts[index], pts[end], color, thickness)

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
        self.draw_dashed_fit(view, debug.get("virtual_fit"), (255, 0, 255), 3)
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
        mode_text = (
            f"mode={debug.get('mode')} err={debug.get('error', 0):.1f} "
            f"d={debug.get('d_term', 0.0):.1f} "
            f"yellow={debug.get('valid_yellow_count', 0)} "
            f"school={int(debug.get('school_zone_detected', False))} "
            f"confirm={debug.get('sharp_confirm', 0)}/{self.WHITE_CURVE_ENTER_FRAMES}"
        )
        if debug.get("virtual_kind"):
            mode_text += f" {debug['virtual_kind']}"
        cv2.putText(
            view,
            mode_text,
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        curve_text = (
            f"geom={debug.get('curve_geometry')} "
            f"head={debug.get('curve_heading', 0.0):.1f}/18,30 "
            f"a={debug.get('curve_quadratic', 0.0):.4f}/.0015,.0040"
        )
        cv2.putText(
            view,
            curve_text,
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
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

            speed = self.apply_speed_recovery_ramp(speed)
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
