#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time
from collections import deque

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

        # Start signal: fixed overhead 3-color light in the upper-center camera area.
        self.START_TL_X_MIN_RATIO = 0.25
        self.START_TL_X_MAX_RATIO = 0.75
        self.START_TL_Y_MIN_RATIO = 0.02
        self.START_TL_Y_MAX_RATIO = 0.42
        self.START_TL_MIN_HOUSING_WIDTH_RATIO = 0.22
        self.START_TL_MAX_HOUSING_WIDTH_RATIO = 0.42
        self.START_TL_MIN_HOUSING_HEIGHT_RATIO = 0.09
        self.START_TL_MAX_HOUSING_HEIGHT_RATIO = 0.22
        self.START_TL_MIN_HOUSING_Y_RATIO = 0.20
        self.START_TL_MAX_HOUSING_Y_RATIO = 0.40
        self.START_TL_MIN_INNER_DARK_RATIO = 0.48
        self.START_TL_MIN_SLOT_DARK_RATIO = 0.30
        self.START_TL_MIN_COLOR_AREA = 40
        self.START_TL_GREEN_CONFIRM_FRAMES = 3
        self.INTERSECTION_TL_X_MIN_RATIO = 0.03
        self.INTERSECTION_TL_X_MAX_RATIO = 0.97
        self.INTERSECTION_TL_Y_MIN_RATIO = 0.02
        self.INTERSECTION_TL_Y_MAX_RATIO = 0.44
        self.INTERSECTION_TL_MIN_COLOR_AREA = 35
        self.INTERSECTION_TL_CONFIRM_FRAMES = 3
        self.INTERSECTION_RIGHT_SLOT_BLACK_RATIO = 0.58

        # Startup cone corridor: the supplied LiDAR viewer uses +y as forward
        # and x as lateral position. Cone navigation is completed once only.
        self.CONE_MIN_RANGE = 0.50
        self.CONE_MAX_RANGE = 5.0
        self.CONE_MIN_FORWARD = 1.0
        self.CONE_MAX_FORWARD = 8.0
        self.CONE_MAX_LATERAL = 5.0
        self.CONE_MAX_CLUSTER_BEAMS = 5
        self.CONE_MAX_CLUSTER_SPAN = 0.60
        self.CONE_MIN_WIDTH = 1.4
        self.CONE_MAX_WIDTH = 7.0
        self.CONE_MAX_PAIR_Y_GAP = 1.6
        self.CONE_ENTRY_FRAMES = 2
        self.CONE_EXIT_FRAMES = 8
        self.CONE_LANE_CONFIRM_FRAMES = 4
        self.CONE_SEARCH_TIMEOUT_FRAMES = 60
        self.CONE_SPEED = 8.0
        self.CONE_KP = 85.0
        self.CONE_MAX_ANGLE = 100.0
        self.CONE_TARGET_ALPHA = 0.25
        self.CONE_TARGET_MAX_STEP = 0.20
        self.CONE_PAIR_MAX_CENTER_JUMP = 1.20
        self.CONE_PAIR_MAX_WIDTH_DELTA = 1.20
        self.CONE_PAIR_CENTER_WEIGHT = 4.0
        self.CONE_PAIR_WIDTH_WEIGHT = 1.5
        self.CONE_LOOKAHEAD_Y = 2.5
        self.CONE_SIDE_X_WEIGHT = 1.5
        self.CONE_SINGLE_MAX_TARGET_JUMP = 0.80

        # Yellow dashed centerline first, white solid-line fallback.
        self.KP = 0.58
        self.YELLOW_CENTER_KP = 0.20
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
        self.WHITE_PAIR_CONFIRM_FRAMES = 2
        self.WHITE_PAIR_CENTER_JUMP_PX = 45.0
        self.WHITE_PAIR_MIN_WIDTH_RATIO = 0.45
        self.WHITE_PAIR_MAX_WIDTH_RATIO = 1.65
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
        self.YELLOW_BASE_CONTINUITY_PX = 80.0
        self.YELLOW_HOLD_FRAMES = 3
        self.YELLOW_STRAIGHT_MAX_ANGLE_DELTA = 10.0
        self.STRAIGHT_ERROR_HISTORY_FRAMES = 5
        self.STRAIGHT_ERROR_SLOW_THRESHOLD = 35.0
        self.STRAIGHT_ERROR_MID_THRESHOLD = 80.0
        self.STRAIGHT_ERROR_SLOW_SPEED = 17.0

        self.yellow_fit = None
        self.white_fit = None
        self.center_fit = None
        self.last_angle = 0.0
        self.last_command_speed = 0.0
        self.prev_yellow_control_error = None
        self.filtered_yellow_error_delta = 0.0
        self.straight_error_history = deque(maxlen=self.STRAIGHT_ERROR_HISTORY_FRAMES)
        self.yellow_hold_remaining = 0
        self.yellow_hold_active = False
        self.yellow_jump_active = False
        self.curve_hold_remaining = 0
        self.sharp_curve_hold_remaining = 0
        self.white_curve_active = False
        self.sharp_curve_confirm_count = 0
        self.straight_confirm_count = 0
        self.sharp_curve_direction = 0
        self.prev_white_curve_candidate_x = None
        self.white_pair_confirm_count = 0
        self.prev_white_pair_center_x = None
        self.fail_count = 0
        self.frame_count = 0

        self.mission_mode = "cone_search"
        self.start_signal_released = False
        self.start_green_confirm_count = 0
        self.start_signal_state = "unknown"
        self.intersection_signal_state = "unknown"
        self.intersection_confirmed_state = "unknown"
        self.intersection_signal_confirm_count = 0
        self.intersection_stop_active = False
        self.intersection_route = "none"
        self.cone_pair_confirm_count = 0
        self.cone_missing_count = 0
        self.yellow_handoff_confirm_count = 0
        self.cone_search_frames = 0
        self.cone_target_x = None
        self.last_cone_width = None
        self.last_cone_pair = None
        self.cone_raw_beam_count = 0
        self.cone_large_cluster_count = 0
        self.cone_debug = {}

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

    def largest_color_blob_area(self, mask):
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        areas = [
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count)
            if self.START_TL_MIN_COLOR_AREA <= stats[index, cv2.CC_STAT_AREA] <= 6000
        ]
        return max(areas) if areas else 0

    def start_signal_slot_circles(self, housing):
        x, y, box_w, box_h = housing
        radius = max(6, int(box_h * 0.34))
        centers = [
            (int(x + box_w * ratio), int(y + box_h * 0.50))
            for ratio in (0.17, 0.50, 0.83)
        ]
        return [(center_x, center_y, radius) for center_x, center_y in centers]

    def detect_start_signal(self, image):
        """Choose the darkest valid 3-lamp rectangle in the upper-center ROI."""
        height, width = image.shape[:2]
        x1 = int(width * self.START_TL_X_MIN_RATIO)
        x2 = int(width * self.START_TL_X_MAX_RATIO)
        y1 = int(height * self.START_TL_Y_MIN_RATIO)
        y2 = int(height * self.START_TL_Y_MAX_RATIO)
        search = image[y1:y2, x1:x2]
        hsv_search = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        gray_search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray_search, 0, 80)
        edges = cv2.Canny(gray_search, 45, 130)

        color_masks = {
            "red": cv2.bitwise_or(
                cv2.inRange(hsv_search, np.array([0, 120, 120]), np.array([10, 255, 255])),
                cv2.inRange(hsv_search, np.array([170, 120, 120]), np.array([180, 255, 255])),
            ),
            "yellow": cv2.inRange(hsv_search, np.array([18, 120, 120]), np.array([42, 255, 255])),
            "green": cv2.inRange(hsv_search, np.array([45, 100, 100]), np.array([95, 255, 255])),
        }

        def rectangle_candidates(mask):
            rectangles = []
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                local_x, local_y, box_w, box_h = cv2.boundingRect(contour)
                if box_w < 35 or box_h < 12:
                    continue
                if box_w > int(width * 0.60) or box_h > int(height * 0.35):
                    continue
                rectangles.append((local_x, local_y, box_w, box_h))
            return rectangles

        dark_rectangles = rectangle_candidates(dark_mask)
        edge_rectangles = rectangle_candidates(edges)

        # The simulator renders the black lamp body as separated dark bands.
        # Merge vertically adjacent bands with almost the same horizontal span.
        merged = list(dark_rectangles)
        for index, first in enumerate(dark_rectangles):
            fx, fy, fw, fh = first
            for second in dark_rectangles[index + 1:]:
                sx, sy, sw, sh = second
                horizontal_overlap = max(0, min(fx + fw, sx + sw) - max(fx, sx))
                vertical_gap = max(sy - (fy + fh), fy - (sy + sh), 0)
                if (
                    horizontal_overlap >= 0.55 * min(fw, sw) and
                    vertical_gap <= max(12, int(0.18 * max(fh, sh)))
                ):
                    left, top = min(fx, sx), min(fy, sy)
                    right, bottom = max(fx + fw, sx + sw), max(fy + fh, sy + sh)
                    merged.append((left, top, right - left, bottom - top))

        def is_housing_shape(rect):
            _, local_y, box_w, box_h = rect
            aspect = box_w / max(box_h, 1)
            return (
                0.20 * width <= box_w <= 0.45 * width and
                0.08 * height <= box_h <= 0.24 * height and
                2.0 <= aspect <= 4.5 and
                0.10 * height <= y1 + local_y <= 0.34 * height
            )

        housing_candidates = []
        for rect in merged:
            if not is_housing_shape(rect):
                continue
            local_x, local_y, box_w, box_h = rect
            patch = dark_mask[local_y:local_y + box_h, local_x:local_x + box_w]
            black_ratio = float(np.mean(patch > 0))
            black_count = int(np.count_nonzero(patch))
            housing_candidates.append((black_ratio, black_count, rect))

        best = max(housing_candidates, key=lambda item: (item[0], item[1])) if housing_candidates else None
        housing = None
        if best is not None:
            local_x, local_y, box_w, box_h = best[2]
            housing = (x1 + local_x, y1 + local_y, box_w, box_h)

        def slot_color_pixels(box):
            scores = {"red": 0, "yellow": 0, "green": 0}
            if box is None:
                return scores
            global_x, global_y, box_w, box_h = box
            radius = max(6, int(box_h * 0.34))
            for color, ratio in zip(("red", "yellow", "green"), (0.17, 0.50, 0.83)):
                slot_mask = np.zeros(hsv_search.shape[:2], dtype=np.uint8)
                center = (int(global_x + box_w * ratio) - x1, int(global_y + box_h * 0.50) - y1)
                cv2.circle(slot_mask, center, radius, 255, -1)
                scores[color] = int(np.count_nonzero(cv2.bitwise_and(color_masks[color], slot_mask)))
            return scores

        scores = slot_color_pixels(housing)
        state = max(scores, key=scores.get)
        if housing is None or scores[state] < self.START_TL_MIN_COLOR_AREA:
            state = "unknown"

        # Keep raw dark and edge boxes visible while tuning the detector.
        debug_rectangles = []
        for local_x, local_y, box_w, box_h in dark_rectangles + edge_rectangles:
            rect = (x1 + local_x, y1 + local_y, box_w, box_h)
            if rect not in debug_rectangles:
                debug_rectangles.append(rect)
        slots = self.start_signal_slot_circles(housing) if housing is not None else []
        return {
            "search_rect": (x1, y1, x2 - x1, y2 - y1),
            "housing_rect": housing,
            "candidates": debug_rectangles,
            "color_rect": housing,
            "slots": slots,
            "candidate_count": len(housing_candidates),
            "scores": scores,
            "state": state,
        }

    def intersection_slot_circles(self, housing):
        x, y, box_w, box_h = housing
        radius = max(5, int(box_h * 0.30))
        return [
            (int(x + box_w * ratio), int(y + box_h * 0.50), radius)
            for ratio in (0.125, 0.375, 0.625, 0.875)
        ]

    def detect_intersection_signal(self, image):
        """Detect the 4-slot intersection signal over a wide upper-camera ROI."""
        height, width = image.shape[:2]
        x1 = int(width * self.INTERSECTION_TL_X_MIN_RATIO)
        x2 = int(width * self.INTERSECTION_TL_X_MAX_RATIO)
        y1 = int(height * self.INTERSECTION_TL_Y_MIN_RATIO)
        y2 = int(height * self.INTERSECTION_TL_Y_MAX_RATIO)
        search = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray, 0, 80)
        edge_mask = cv2.Canny(gray, 45, 130)

        color_masks = {
            "red": cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255])),
                cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255])),
            ),
            "yellow": cv2.inRange(hsv, np.array([18, 120, 120]), np.array([42, 255, 255])),
            "green": cv2.inRange(hsv, np.array([45, 100, 100]), np.array([95, 255, 255])),
        }

        def rectangles(mask):
            result = []
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                local_x, local_y, box_w, box_h = cv2.boundingRect(contour)
                if box_w >= 35 and box_h >= 12 and box_w <= int(width * 0.75) and box_h <= int(height * 0.30):
                    result.append((local_x, local_y, box_w, box_h))
            return result

        dark_rects = rectangles(dark_mask)
        edge_rects = rectangles(edge_mask)
        merged = list(dark_rects)
        for index, first in enumerate(dark_rects):
            fx, fy, fw, fh = first
            for second in dark_rects[index + 1:]:
                sx, sy, sw, sh = second
                overlap = max(0, min(fx + fw, sx + sw) - max(fx, sx))
                vertical_gap = max(sy - (fy + fh), fy - (sy + sh), 0)
                if overlap >= 0.50 * min(fw, sw) and vertical_gap <= max(14, int(0.22 * max(fh, sh))):
                    left, top = min(fx, sx), min(fy, sy)
                    right, bottom = max(fx + fw, sx + sw), max(fy + fh, sy + sh)
                    merged.append((left, top, right - left, bottom - top))

        candidates = []
        for local_x, local_y, box_w, box_h in merged:
            aspect = box_w / max(box_h, 1)
            global_y = y1 + local_y
            if not (
                0.24 * width <= box_w <= 0.75 * width and
                0.07 * height <= box_h <= 0.24 * height and
                2.8 <= aspect <= 7.0 and
                0.06 * height <= global_y <= 0.36 * height
            ):
                continue
            patch = dark_mask[local_y:local_y + box_h, local_x:local_x + box_w]
            black_ratio = float(np.mean(patch > 0))
            black_count = int(np.count_nonzero(patch))
            candidates.append((black_ratio, black_count, (local_x, local_y, box_w, box_h)))

        best = max(candidates, key=lambda item: (item[0], item[1])) if candidates else None
        housing = None
        if best is not None:
            local_x, local_y, box_w, box_h = best[2]
            housing = (x1 + local_x, y1 + local_y, box_w, box_h)

        slot_scores = [
            {"red": 0, "yellow": 0, "green": 0, "black_ratio": 0.0}
            for _ in range(4)
        ]
        slots = self.intersection_slot_circles(housing) if housing is not None else []
        for index, (center_x, center_y, radius) in enumerate(slots):
            slot_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            local_center = (center_x - x1, center_y - y1)
            cv2.circle(slot_mask, local_center, radius, 255, -1)
            pixels = gray[slot_mask > 0]
            slot_scores[index]["black_ratio"] = float(np.mean(pixels < 100)) if len(pixels) else 1.0
            for color, mask in color_masks.items():
                slot_scores[index][color] = int(np.count_nonzero(cv2.bitwise_and(mask, slot_mask)))

        scores = {
            "red": slot_scores[0]["red"],
            "yellow": slot_scores[1]["yellow"],
            "green_left": slot_scores[2]["green"],
            "green_straight": slot_scores[3]["green"],
            "right_black": slot_scores[3]["black_ratio"],
        }
        state = "unknown"
        if housing is not None:
            # Left-turn phase: slot 1 red + slot 3 green, with slot 4 dark.
            # The green-arrow combination must override the red stop light.
            left_arrow_active = (
                scores["green_left"] >= self.INTERSECTION_TL_MIN_COLOR_AREA and
                scores["right_black"] >= self.INTERSECTION_RIGHT_SLOT_BLACK_RATIO
            )
            if left_arrow_active:
                state = "left_green"
            elif scores["red"] >= self.INTERSECTION_TL_MIN_COLOR_AREA:
                state = "red"
            elif scores["yellow"] >= self.INTERSECTION_TL_MIN_COLOR_AREA:
                state = "yellow"
            elif scores["green_straight"] >= self.INTERSECTION_TL_MIN_COLOR_AREA:
                state = "straight_green"
            elif scores["green_left"] >= self.INTERSECTION_TL_MIN_COLOR_AREA:
                state = "straight_green"
        debug_rectangles = []
        for local_x, local_y, box_w, box_h in dark_rects + edge_rects:
            rect = (x1 + local_x, y1 + local_y, box_w, box_h)
            if rect not in debug_rectangles:
                debug_rectangles.append(rect)
        return {
            "search_rect": (x1, y1, x2 - x1, y2 - y1),
            "housing_rect": housing,
            "candidates": debug_rectangles,
            "slots": slots,
            "candidate_count": len(candidates),
            "scores": scores,
            "slot_scores": slot_scores,
            "state": state,
        }

    def update_intersection_signal(self, signal):
        state = signal["state"]
        self.intersection_signal_state = state
        if state == "unknown":
            self.intersection_signal_confirm_count = 0
            return

        if state == self.intersection_confirmed_state:
            self.intersection_signal_confirm_count += 1
        else:
            self.intersection_confirmed_state = state
            self.intersection_signal_confirm_count = 1

        if self.intersection_signal_confirm_count < self.INTERSECTION_TL_CONFIRM_FRAMES:
            return
        self.intersection_stop_active = state in ("red", "yellow")
        if state == "left_green":
            self.intersection_route = "left"
        elif state == "straight_green":
            self.intersection_route = "straight"

    def show_intersection_traffic_debug(self, image, signal):
        if not self.show_debug:
            return
        view = image.copy()
        sx, sy, sw, sh = signal["search_rect"]
        cv2.rectangle(view, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
        for hx, hy, hw, hh in signal.get("candidates", []):
            cv2.rectangle(view, (hx, hy), (hx + hw, hy + hh), (0, 165, 255), 1)
        if signal["housing_rect"] is not None:
            hx, hy, hw, hh = signal["housing_rect"]
            cv2.rectangle(view, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), 2)
        for index, (center_x, center_y, radius) in enumerate(signal.get("slots", [])):
            cv2.circle(view, (center_x, center_y), radius, (0, 255, 255), 1)
            cv2.putText(view, str(index + 1), (center_x - 4, center_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        scores = signal["scores"]
        cv2.putText(
            view,
            f"intersection={signal['state']} confirmed={self.intersection_confirmed_state} "
            f"stop={int(self.intersection_stop_active)} route={self.intersection_route}",
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2,
        )
        cv2.putText(
            view,
            f"R={scores['red']} Y={scores['yellow']} GL={scores['green_left']} "
            f"GS={scores['green_straight']} right_black={scores['right_black']:.2f} "
            f"confirm={self.intersection_signal_confirm_count}/{self.INTERSECTION_TL_CONFIRM_FRAMES}",
            (15, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
        )
        cv2.imshow("traffic light debug", view)

    def update_start_signal(self, signal):
        self.start_signal_state = signal["state"]
        if self.start_signal_released:
            return
        if signal["state"] == "green":
            self.start_green_confirm_count += 1
            if self.start_green_confirm_count >= self.START_TL_GREEN_CONFIRM_FRAMES:
                self.start_signal_released = True
                self.get_logger().info("start signal green confirmed: cone mission enabled")
        else:
            self.start_green_confirm_count = 0

    def show_traffic_debug(self, image, signal):
        if not self.show_debug:
            return
        view = image.copy()
        sx, sy, sw, sh = signal["search_rect"]
        cv2.rectangle(view, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 2)
        for hx, hy, hw, hh in signal.get("candidates", []):
            cv2.rectangle(view, (hx, hy), (hx + hw, hy + hh), (0, 165, 255), 2)
        if signal["housing_rect"] is not None:
            hx, hy, hw, hh = signal["housing_rect"]
            cv2.rectangle(view, (hx, hy), (hx + hw, hy + hh), (255, 0, 255), 2)
        for center_x, center_y, radius in signal.get("slots", []):
            cv2.circle(view, (center_x, center_y), radius, (0, 255, 255), 1)
        scores = signal["scores"]
        cv2.putText(
            view,
            f"start={signal['state']} released={int(self.start_signal_released)} "
            f"green={self.start_green_confirm_count}/{self.START_TL_GREEN_CONFIRM_FRAMES}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        )
        cv2.putText(
            view,
            f"R={scores['red']} Y={scores['yellow']} G={scores['green']} "
            f"rects={signal.get('candidate_count', 0)}",
            (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        )
        cv2.imshow("traffic light debug", view)

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

    def update_white_pair(self, white_lanes, white_fits, width, height):
        """Return a stable actual left/right white pair enclosing the vehicle."""
        y_reference = int(height * 0.64)
        candidates = []
        for lane, fit in zip(white_lanes, white_fits):
            if fit is None:
                continue
            x_reference = float(np.polyval(fit, y_reference))
            if -width * 0.10 <= x_reference <= width * 1.10:
                candidates.append((x_reference, lane, fit))

        left = [candidate for candidate in candidates if candidate[0] < width / 2.0]
        right = [candidate for candidate in candidates if candidate[0] > width / 2.0]
        if not left or not right:
            self.white_pair_confirm_count = 0
            self.prev_white_pair_center_x = None
            return None

        # Pick the two physical boundaries immediately surrounding vehicle center.
        left_x, left_lane, left_fit = max(left, key=lambda item: item[0])
        right_x, right_lane, right_fit = min(right, key=lambda item: item[0])
        lane_width = right_x - left_x
        min_width = self.LANE_WIDTH_PX * self.WHITE_PAIR_MIN_WIDTH_RATIO
        max_width = self.LANE_WIDTH_PX * self.WHITE_PAIR_MAX_WIDTH_RATIO
        center_x = 0.5 * (left_x + right_x)

        if not (min_width <= lane_width <= max_width):
            self.white_pair_confirm_count = 0
            self.prev_white_pair_center_x = None
            return None

        continuous = (
            self.prev_white_pair_center_x is None or
            abs(center_x - self.prev_white_pair_center_x) <= self.WHITE_PAIR_CENTER_JUMP_PX
        )
        if not continuous:
            self.white_pair_confirm_count = 0
            self.prev_white_pair_center_x = center_x
            return None

        self.white_pair_confirm_count += 1
        self.prev_white_pair_center_x = center_x
        if self.white_pair_confirm_count < self.WHITE_PAIR_CONFIRM_FRAMES:
            return None

        return {
            "left_lane": left_lane,
            "right_lane": right_lane,
            "left_fit": left_fit,
            "right_fit": right_fit,
            "center_fit": 0.5 * (left_fit + right_fit),
            "width": lane_width,
        }

    def find_best_lane(self, lane_mask, min_peak, previous_fit, top_y, bottom_y, width, height,
                       min_pixels=None, require_bottom=True, bases_override=None):
        bases = (
            bases_override if bases_override is not None else
            self.find_lane_bases(lane_mask, min_peak, previous_fit)
        )
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

    def find_stable_yellow_lane(self, yellow_mask, top_y, bottom_y, width, height):
        """Prefer yellow peaks near the previous fit and briefly hold on a jump."""
        previous_fit = self.yellow_fit
        if previous_fit is None:
            result = self.find_best_lane(
                yellow_mask, self.YELLOW_MIN_PEAK, None, top_y, bottom_y, width, height,
                min_pixels=self.YELLOW_MIN_FIT_PIXELS, require_bottom=False,
            )
            self.yellow_hold_active = False
            return (*result, False)

        y_reference = int(height * 0.64)
        predicted_base = int(np.clip(
            np.polyval(previous_fit, bottom_y), 0, width - 1
        ))
        raw_bases = self.find_lane_bases(yellow_mask, self.YELLOW_MIN_PEAK, None)
        nearby_bases = [
            base for base in raw_bases
            if abs(base - predicted_base) <= self.YELLOW_BASE_CONTINUITY_PX
        ]

        def hold_previous():
            if self.yellow_hold_remaining <= 0:
                return None
            self.yellow_hold_remaining -= 1
            self.yellow_hold_active = True
            held_fit = previous_fit.copy()
            held_side = "left" if np.polyval(held_fit, y_reference) < width / 2.0 else "right"
            return None, held_fit, held_side, [], [held_fit], True

        if raw_bases and not nearby_bases:
            if not self.yellow_jump_active:
                self.yellow_jump_active = True
                self.yellow_hold_remaining = self.YELLOW_HOLD_FRAMES
            held = hold_previous()
            if held is not None:
                return held
            # The same peak remained far for the full hold period. Re-acquire it
            # instead of indefinitely preserving an obsolete previous fit.
            bases = raw_bases
        else:
            bases = nearby_bases if nearby_bases else [predicted_base]
        result = self.find_best_lane(
            yellow_mask, self.YELLOW_MIN_PEAK, previous_fit, top_y, bottom_y, width, height,
            min_pixels=self.YELLOW_MIN_FIT_PIXELS, require_bottom=False,
            bases_override=bases,
        )
        lane, fit, side, lanes, fits = result
        if fit is not None:
            self.yellow_hold_remaining = 0
            self.yellow_hold_active = False
            self.yellow_jump_active = False
            return lane, fit, side, lanes, fits, False

        held = hold_previous()
        if held is not None:
            return held
        self.yellow_hold_active = False
        return lane, fit, side, lanes, fits, False

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
        # A large heading alone can be a straight-road alignment error.
        # White-curve mode requires measurable lane curvature as well.
        raw_sharp = (
            abs(heading) > self.WHITE_CURVE_COMBINED_HEADING_DEG and
            quadratic > self.WHITE_CURVE_COMBINED_QUADRATIC
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

    def p_control(self, error, mode, max_angle_delta=None):
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
        allowed_delta = self.MAX_ANGLE_DELTA if max_angle_delta is None else max_angle_delta
        delta = float(np.clip(raw_angle - self.last_angle, -allowed_delta, allowed_delta))
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

    def apply_straight_error_speed(self, requested_speed, error, mode, yellow_straight):
        """Slow a straight yellow-center pass only while its lateral error is large."""
        if mode != "yellow_center" or not yellow_straight:
            self.straight_error_history.clear()
            return requested_speed, 0.0, "off"

        self.straight_error_history.append(abs(error))
        average_error = float(np.mean(self.straight_error_history))
        if average_error > self.STRAIGHT_ERROR_MID_THRESHOLD:
            return min(requested_speed, self.MID_SPEED), average_error, "mid"
        if average_error > self.STRAIGHT_ERROR_SLOW_THRESHOLD:
            return (
                min(requested_speed, self.STRAIGHT_ERROR_SLOW_SPEED),
                average_error,
                "slow",
            )
        return requested_speed, average_error, "base"

    def decide_control(self, image):
        height, width = image.shape[:2]
        yellow_mask, white_mask, roi_polygon, top_y, bottom_y = self.make_lane_masks(image)

        (yellow_lane, yellow_fit, yellow_side, yellow_lanes, yellow_fits,
         yellow_held) = self.find_stable_yellow_lane(
            yellow_mask, top_y, bottom_y, width, height)
        white_lane, white_fit, white_side, white_lanes, white_fits = self.find_best_lane(
            white_mask, self.WHITE_MIN_PEAK, self.white_fit, top_y, bottom_y, width, height,
            min_pixels=self.MIN_FIT_PIXELS, require_bottom=False)
        white_pair = self.update_white_pair(white_lanes, white_fits, width, height)

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
            "yellow_held": yellow_held,
            "yellow_hold_remaining": self.yellow_hold_remaining,
            "white_pair": white_pair,
            "white_pair_confirm": self.white_pair_confirm_count,
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
                if white_pair is not None:
                    self.center_fit = white_pair["center_fit"]
                    mode = "white_pair_curve"
                else:
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
            self.straight_error_history.clear()
            self.yellow_hold_remaining = 0
            self.yellow_hold_active = False
            self.yellow_jump_active = False
            self.curve_hold_remaining = 0
            self.sharp_curve_hold_remaining = 0
            self.white_curve_active = False
            self.sharp_curve_confirm_count = 0
            self.straight_confirm_count = 0
            self.sharp_curve_direction = 0
            self.prev_white_curve_candidate_x = None
            self.white_pair_confirm_count = 0
            self.prev_white_pair_center_x = None
            return 0.0, self.LOST_SPEED, debug

        virtual_fit, virtual_kind = self.make_virtual_opposite_fit(
            yellow_fit, yellow_side, white_fit, white_side)

        error, target_x, y_target = self.calculate_error(self.center_fit, width, height)
        yellow_straight = (
            mode == "yellow_center" and
            curve_quadratic <= self.CURVE_QUADRATIC
        )
        angle_delta_limit = (
            self.YELLOW_STRAIGHT_MAX_ANGLE_DELTA if yellow_straight else None
        )
        angle, d_term = self.p_control(error, mode, angle_delta_limit)
        speed = self.select_speed(angle, mode)
        speed, straight_error_average, straight_speed_mode = self.apply_straight_error_speed(
            speed, error, mode, yellow_straight)

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
            "yellow_held": yellow_held,
            "yellow_hold_remaining": self.yellow_hold_remaining,
            "yellow_straight_limiter": yellow_straight,
            "straight_error_average": straight_error_average,
            "straight_speed_mode": straight_speed_mode,
            "white_pair": white_pair,
            "white_pair_confirm": self.white_pair_confirm_count,
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

    def lidar_cone_clusters(self):
        """Return LiDAR obstacle clusters in the same coordinates as lidar_viewer.

        The simulator scan is ordered with index zero at the forward direction.
        The visualizer maps it to x=lateral and y=forward using this transform.
        """
        if not self.lidar_ranges:
            self.cone_raw_beam_count = 0
            self.cone_large_cluster_count = 0
            return []

        ranges = np.asarray(self.lidar_ranges, dtype=np.float32)
        count = len(ranges)
        if count == 0:
            self.cone_raw_beam_count = 0
            self.cone_large_cluster_count = 0
            return []

        indices = np.arange(count, dtype=np.float32)
        angles = np.deg2rad(indices * (360.0 / count) - 90.0)
        x_values = -ranges * np.cos(angles)
        y_values = -ranges * np.sin(angles)
        valid = (
            np.isfinite(ranges) &
            (ranges >= self.CONE_MIN_RANGE) &
            (ranges <= self.CONE_MAX_RANGE) &
            (y_values >= self.CONE_MIN_FORWARD) &
            (y_values <= self.CONE_MAX_FORWARD) &
            (np.abs(x_values) <= self.CONE_MAX_LATERAL)
        )
        self.cone_raw_beam_count = int(np.count_nonzero(valid))

        clusters = []
        self.cone_large_cluster_count = 0
        active = []
        previous_index = None
        previous_point = None

        def finish_cluster(points):
            if not points:
                return
            points = np.asarray(points, dtype=np.float32)
            span = float(np.max(np.linalg.norm(points - points[0], axis=1)))
            if (len(points) > self.CONE_MAX_CLUSTER_BEAMS or
                    span > self.CONE_MAX_CLUSTER_SPAN):
                self.cone_large_cluster_count += 1
                return
            clusters.append({
                "x": float(np.median(points[:, 0])),
                "y": float(np.median(points[:, 1])),
                "beams": len(points),
                "span": span,
            })

        for index in np.flatnonzero(valid):
            point = (x_values[index], y_values[index])
            adjacent = (
                previous_index is not None and
                index - previous_index <= 2 and
                np.hypot(point[0] - previous_point[0], point[1] - previous_point[1]) <= 0.65
            )
            if active and not adjacent:
                finish_cluster(active)
                active = []
            active.append(point)
            previous_index = index
            previous_point = point
        finish_cluster(active)
        return clusters

    def select_cone_side(self, candidates, side):
        """Pick one stable cone candidate for one vehicle side."""
        if not candidates:
            return None

        expected_x = None
        if self.cone_target_x is not None and self.last_cone_width is not None:
            expected_x = self.cone_target_x + side * self.last_cone_width * 0.5

        def score(point):
            lookahead_error = abs(point["y"] - self.CONE_LOOKAHEAD_Y)
            lateral_error = abs(point["x"] - expected_x) if expected_x is not None else 0.0
            return lookahead_error + self.CONE_SIDE_X_WEIGHT * lateral_error

        return min(candidates, key=score)

    def make_single_side_corridor(self, point, side, clusters):
        """Use one boundary only when its inferred center stays near the last path."""
        if point is None or self.last_cone_width is None or self.cone_target_x is None:
            return None

        if side < 0:
            left_point = point
            right_point = {"x": point["x"] + self.last_cone_width, "y": point["y"], "beams": 0}
            source = "left_only"
        else:
            right_point = point
            left_point = {"x": point["x"] - self.last_cone_width, "y": point["y"], "beams": 0}
            source = "right_only"

        target_x = 0.5 * (left_point["x"] + right_point["x"])
        center_jump = abs(target_x - self.cone_target_x)
        if center_jump > self.CONE_SINGLE_MAX_TARGET_JUMP:
            self.cone_debug["rejected_single"] += 1
            return None

        self.cone_debug.update({
            "source": source,
            "width": self.last_cone_width,
            "raw_target_x": target_x,
            "center_jump": center_jump,
        })
        return {
            "left": left_point,
            "right": right_point,
            "width": self.last_cone_width,
            "target_x": target_x,
            "source": source,
            "clusters": clusters,
        }

    def find_cone_corridor(self):
        """Choose one left cone and one right cone before validating a corridor."""
        clusters = self.lidar_cone_clusters()
        left_candidates = [point for point in clusters if point["x"] < -0.25]
        right_candidates = [point for point in clusters if point["x"] > 0.25]
        left_point = self.select_cone_side(left_candidates, -1)
        right_point = self.select_cone_side(right_candidates, +1)
        previous_target_x = self.cone_target_x
        previous_width = self.last_cone_width

        self.cone_debug = {
            "raw_beams": self.cone_raw_beam_count,
            "large_rejected": self.cone_large_cluster_count,
            "clusters": clusters,
            "left_count": len(left_candidates),
            "right_count": len(right_candidates),
            "pair_count": 0,
            "rejected_width": 0,
            "rejected_y_gap": 0,
            "rejected_continuity": 0,
            "rejected_single": 0,
            "source": "none",
        }

        if left_point is not None and right_point is not None:
            width = right_point["x"] - left_point["x"]
            y_gap = abs(right_point["y"] - left_point["y"])
            target_x = 0.5 * (left_point["x"] + right_point["x"])
            center_jump = abs(target_x - previous_target_x) if previous_target_x is not None else 0.0
            width_delta = abs(width - previous_width) if previous_width is not None else 0.0
            valid_pair = True
            if not (self.CONE_MIN_WIDTH <= width <= self.CONE_MAX_WIDTH):
                self.cone_debug["rejected_width"] += 1
                valid_pair = False
            if y_gap > self.CONE_MAX_PAIR_Y_GAP:
                self.cone_debug["rejected_y_gap"] += 1
                valid_pair = False
            if previous_target_x is not None and center_jump > self.CONE_PAIR_MAX_CENTER_JUMP:
                self.cone_debug["rejected_continuity"] += 1
                valid_pair = False
            if previous_width is not None and width_delta > self.CONE_PAIR_MAX_WIDTH_DELTA:
                self.cone_debug["rejected_continuity"] += 1
                valid_pair = False

            if valid_pair:
                self.cone_debug["pair_count"] = 1
                self.last_cone_width = width if previous_width is None else 0.80 * previous_width + 0.20 * width
                pair = {
                    "left": left_point,
                    "right": right_point,
                    "width": width,
                    "target_x": target_x,
                    "source": "pair",
                    "clusters": clusters,
                }
                self.last_cone_pair = pair
                self.cone_debug.update({
                    "source": "pair",
                    "width": width,
                    "raw_target_x": target_x,
                    "center_jump": center_jump,
                    "width_delta": width_delta,
                })
                return pair

        right_only = self.make_single_side_corridor(right_point, +1, clusters)
        if right_only is not None:
            return right_only
        left_only = self.make_single_side_corridor(left_point, -1, clusters)
        if left_only is not None:
            return left_only
        return None

    def reset_lane_control_state(self):
        self.prev_yellow_control_error = None
        self.filtered_yellow_error_delta = 0.0
        self.last_angle = 0.0

    def update_mission_mode(self, yellow_fit):
        """Advance from the startup cones to lane following only after yellow confirmation."""
        corridor = self.find_cone_corridor()
        has_full_cone_pair = corridor is not None and corridor["source"] == "pair"
        has_cone_guidance = corridor is not None
        yellow_visible = yellow_fit is not None

        if self.mission_mode == "cone_search":
            self.cone_search_frames += 1
            if has_full_cone_pair:
                self.cone_pair_confirm_count += 1
                if self.cone_pair_confirm_count >= self.CONE_ENTRY_FRAMES:
                    self.mission_mode = "cone_nav"
                    self.cone_missing_count = 0
                    self.yellow_handoff_confirm_count = 0
            else:
                self.cone_pair_confirm_count = 0
                if yellow_visible and self.cone_search_frames >= self.CONE_SEARCH_TIMEOUT_FRAMES:
                    self.yellow_handoff_confirm_count += 1
                    if self.yellow_handoff_confirm_count >= self.CONE_LANE_CONFIRM_FRAMES:
                        self.mission_mode = "lane_follow"
                        self.reset_lane_control_state()
                else:
                    self.yellow_handoff_confirm_count = 0

        elif self.mission_mode == "cone_nav":
            if has_cone_guidance:
                self.cone_missing_count = 0
                self.yellow_handoff_confirm_count = 0
            else:
                self.mission_mode = "cone_exit_confirm"
                self.cone_missing_count = 1
                self.yellow_handoff_confirm_count = 1 if yellow_visible else 0

        elif self.mission_mode == "cone_exit_confirm":
            if has_cone_guidance:
                self.mission_mode = "cone_nav"
                self.cone_missing_count = 0
                self.yellow_handoff_confirm_count = 0
            else:
                self.cone_missing_count += 1
                self.yellow_handoff_confirm_count = (
                    self.yellow_handoff_confirm_count + 1 if yellow_visible else 0
                )
                if (self.cone_missing_count >= self.CONE_EXIT_FRAMES and
                        self.yellow_handoff_confirm_count >= self.CONE_LANE_CONFIRM_FRAMES):
                    self.mission_mode = "lane_follow"
                    self.cone_target_x = None
                    self.reset_lane_control_state()

        return corridor

    def cone_control(self, corridor):
        """Steer toward the LiDAR corridor midpoint while limiting target jumps."""
        if corridor is None:
            return self.last_angle, min(self.CONE_SPEED, 6.0)

        raw_target_x = float(corridor["target_x"])
        if self.cone_target_x is None:
            self.cone_target_x = raw_target_x
        else:
            delta = float(np.clip(
                raw_target_x - self.cone_target_x,
                -self.CONE_TARGET_MAX_STEP,
                self.CONE_TARGET_MAX_STEP,
            ))
            self.cone_target_x += self.CONE_TARGET_ALPHA * delta

        raw_angle = float(np.clip(
            self.CONE_KP * self.cone_target_x,
            -self.CONE_MAX_ANGLE,
            self.CONE_MAX_ANGLE,
        ))
        angle_delta = float(np.clip(raw_angle - self.last_angle, -80.0, 80.0))
        angle = float(np.clip(self.last_angle + angle_delta, -self.MAX_ANGLE, self.MAX_ANGLE))
        self.last_angle = angle
        self.prev_yellow_control_error = None
        self.filtered_yellow_error_delta = 0.0
        speed = self.CONE_SPEED if corridor["source"] == "pair" else min(self.CONE_SPEED, 6.0)
        return angle, speed

    def front_obstacle_distance(self):
        if not self.lidar_ranges:
            return math.inf

        ranges = np.array(self.lidar_ranges, dtype=np.float32)
        if not np.any(np.isfinite(ranges)):
            return math.inf

        # The supplied LiDAR viewer uses index 0 as forward; its forward wedge
        # wraps around the beginning/end of the scan rather than the midpoint.
        half_width = max(8, len(ranges) // 36)
        indices = np.concatenate((
            np.arange(0, half_width + 1),
            np.arange(len(ranges) - half_width, len(ranges)),
        ))
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

    def draw_cone_lidar_inset(self, view, cone_debug, corridor):
        """Draw a compact top-down LiDAR view in the existing camera window."""
        height, width = view.shape[:2]
        map_width = min(220, max(160, width // 3))
        map_height = min(175, max(125, height // 3))
        left = width - map_width - 10
        top = height - map_height - 10
        cv2.rectangle(view, (left, top), (left + map_width, top + map_height), (35, 35, 35), -1)
        cv2.rectangle(view, (left, top), (left + map_width, top + map_height), (0, 255, 255), 1)

        origin_x = left + map_width // 2
        origin_y = top + map_height - 10
        scale = min((map_width - 20) / (2.0 * self.CONE_MAX_LATERAL),
                    (map_height - 20) / self.CONE_MAX_FORWARD)
        cv2.line(view, (origin_x, top + 5), (origin_x, origin_y), (90, 90, 90), 1)
        cv2.line(view, (left + 5, origin_y), (left + map_width - 5, origin_y), (90, 90, 90), 1)
        cv2.arrowedLine(view, (origin_x, origin_y), (origin_x, origin_y - 25), (0, 0, 255), 2)

        def point_to_pixel(point):
            px = int(np.clip(origin_x + point["x"] * scale, left + 3, left + map_width - 3))
            py = int(np.clip(origin_y - point["y"] * scale, top + 3, origin_y))
            return px, py

        for point in cone_debug.get("clusters", []):
            cv2.circle(view, point_to_pixel(point), 4, (210, 210, 210), -1)

        if corridor is not None:
            left_point = point_to_pixel(corridor["left"])
            right_point = point_to_pixel(corridor["right"])
            target = {
                "x": corridor["target_x"],
                "y": 0.5 * (corridor["left"]["y"] + corridor["right"]["y"]),
            }
            target_point = point_to_pixel(target)
            cv2.circle(view, left_point, 6, (0, 255, 0), -1)
            cv2.circle(view, right_point, 6, (0, 165, 255), -1)
            cv2.circle(view, target_point, 5, (0, 0, 255), -1)
            cv2.line(view, (origin_x, origin_y), target_point, (0, 0, 255), 2)

        cv2.putText(view, "LiDAR cone map", (left + 6, top + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

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
        white_pair = debug.get("white_pair")
        if white_pair is not None:
            self.draw_fit(view, white_pair["left_fit"], (255, 255, 0), 2)
            self.draw_fit(view, white_pair["right_fit"], (255, 255, 0), 2)
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
            f"mission={debug.get('mission_mode', 'lane_follow')} lane={debug.get('mode')} "
            f"err={debug.get('error', 0):.1f} d={debug.get('d_term', 0.0):.1f} "
            f"yellow={debug.get('valid_yellow_count', 0)} "
            f"hold={int(debug.get('yellow_held', False))}/"
            f"{debug.get('yellow_hold_remaining', 0)} "
            f"lim={int(debug.get('yellow_straight_limiter', False))} "
            f"erravg={debug.get('straight_error_average', 0.0):.1f} "
            f"spd={debug.get('straight_speed_mode', 'off')} "
            f"wpair={debug.get('white_pair_confirm', 0)}/{self.WHITE_PAIR_CONFIRM_FRAMES}"
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
        if debug.get("mission_mode", "") in ("cone_search", "cone_nav", "cone_exit_confirm"):
            cone_debug = debug.get("cone_debug", {})
            curve_text = (
                f"cone={debug.get('cone_source', 'none')} "
                f"raw/filt={cone_debug.get('raw_target_x', 0.0):+.2f}/"
                f"{debug.get('cone_target_x', 0.0):+.2f}m "
                f"pair={debug.get('cone_pair_confirm', 0)}/{self.CONE_ENTRY_FRAMES} "
                f"exit={debug.get('cone_missing', 0)}/{self.CONE_EXIT_FRAMES} "
                f"yellow={debug.get('yellow_handoff_confirm', 0)}/{self.CONE_LANE_CONFIRM_FRAMES}"
            )
            diagnostic_text = (
                f"lidar beams={cone_debug.get('raw_beams', 0)} large="
                f"{cone_debug.get('large_rejected', 0)} clusters="
                f"{len(cone_debug.get('clusters', []))} L/R="
                f"{cone_debug.get('left_count', 0)}/{cone_debug.get('right_count', 0)} "
                f"pairs={cone_debug.get('pair_count', 0)} reject(w/dy/c)="
                f"{cone_debug.get('rejected_width', 0)}/"
                f"{cone_debug.get('rejected_y_gap', 0)}/"
                f"{cone_debug.get('rejected_continuity', 0)} "
                f"single_reject={cone_debug.get('rejected_single', 0)} "
                f"width={cone_debug.get('width', 0.0):.2f}"
            )
            self.draw_cone_lidar_inset(view, cone_debug, debug.get("cone_corridor"))
        else:
            diagnostic_text = ""
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
        if diagnostic_text:
            cv2.putText(
                view,
                diagnostic_text,
                (15, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
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

            lane_angle, lane_speed, debug = self.decide_control(self.image)
            start_signal = self.detect_start_signal(self.image)
            self.update_start_signal(start_signal)
            intersection_signal = None
            if self.start_signal_released:
                intersection_signal = self.detect_intersection_signal(self.image)
                self.update_intersection_signal(intersection_signal)

            if not self.start_signal_released:
                corridor = None
                debug.update({
                    "mission_mode": "start_wait_signal",
                    "cone_pair_confirm": 0,
                    "cone_missing": 0,
                    "yellow_handoff_confirm": 0,
                    "cone_source": "none",
                    "cone_target_x": 0.0,
                    "cone_corridor": None,
                    "cone_debug": {},
                    "start_signal": start_signal,
                    "intersection_signal": None,
                })
                angle, speed = 0.0, 0.0
            else:
                corridor = self.update_mission_mode(debug.get("yellow_fit"))
                debug.update({
                    "mission_mode": self.mission_mode,
                    "cone_pair_confirm": self.cone_pair_confirm_count,
                    "cone_missing": self.cone_missing_count,
                    "yellow_handoff_confirm": self.yellow_handoff_confirm_count,
                    "cone_source": corridor["source"] if corridor is not None else "none",
                    "cone_target_x": self.cone_target_x if self.cone_target_x is not None else 0.0,
                    "cone_corridor": corridor,
                    "cone_debug": dict(self.cone_debug),
                    "start_signal": start_signal,
                    "intersection_signal": intersection_signal,
                })

                if self.mission_mode in ("cone_nav", "cone_exit_confirm"):
                    angle, speed = self.cone_control(corridor)
                elif self.mission_mode == "cone_search":
                    # Wait for a stable left/right pair before leaving the start line.
                    angle, speed = 0.0, 0.0
                else:
                    angle, speed = lane_angle, lane_speed

            # A confirmed red/yellow intersection signal takes priority over lane control.
            if self.start_signal_released and self.intersection_stop_active:
                angle, speed = 0.0, 0.0

            debug["cone_target_x"] = self.cone_target_x if self.cone_target_x is not None else 0.0
            front_dist = self.front_obstacle_distance()

            if self.frame_count % 15 == 0 and self.mission_mode in ("cone_search", "cone_nav", "cone_exit_confirm"):
                cone_debug = debug["cone_debug"]
                self.get_logger().info(
                    "cone state=%s source=%s beams=%d large=%d clusters=%d L/R=%d/%d pairs=%d "
                    "reject_w/dy/c=%d/%d/%d raw_x=%+.2f filt_x=%+.2f jump=%+.2f width_d=%+.2f exit=%d yellow=%d" % (
                        self.mission_mode,
                        debug["cone_source"],
                        cone_debug.get("raw_beams", 0),
                        cone_debug.get("large_rejected", 0),
                        len(cone_debug.get("clusters", [])),
                        cone_debug.get("left_count", 0),
                        cone_debug.get("right_count", 0),
                        cone_debug.get("pair_count", 0),
                        cone_debug.get("rejected_width", 0),
                        cone_debug.get("rejected_y_gap", 0),
                        cone_debug.get("rejected_continuity", 0),
                        cone_debug.get("raw_target_x", 0.0),
                        debug["cone_target_x"],
                        cone_debug.get("center_jump", 0.0),
                        cone_debug.get("width_delta", 0.0),
                        self.cone_missing_count,
                        self.yellow_handoff_confirm_count,
                    )
                )

            if front_dist < 0.8:
                angle, speed = 0.0, 0.0
            elif front_dist < 1.5:
                speed = min(speed, 8.0)

            speed = self.apply_speed_recovery_ramp(speed)
            self.drive(angle=angle, speed=speed)

            self.frame_count += 1
            if self.frame_count % 2 == 0:
                self.show_debug_image(self.image, debug, angle, speed, front_dist)
                if self.start_signal_released and intersection_signal is not None:
                    self.show_intersection_traffic_debug(self.image, intersection_signal)
                else:
                    self.show_traffic_debug(self.image, start_signal)

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
