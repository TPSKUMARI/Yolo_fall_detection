#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

class RealtimeFallDetector:
    def __init__(self, model_path, conf_threshold=0.5, camera_index=0, cooldown_frames=90):
        """
        Initialize real-time fall detector
        
        Args:
            model_path (str): Path to the YOLOv8 model (.pt file)
            conf_threshold (float): Confidence threshold for detections
            camera_index (int): Camera device index (0 for default webcam)
            cooldown_frames (int): Number of frames to wait before detecting another fall
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.camera_index = camera_index
        
        # FPS calculation
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()
        
        # Detection statistics
        self.fall_count = 0
        self.total_detections = 0
        
        # Alert system
        self.alert_active = False
        self.alert_cooldown = 0
        self.cooldown_frames = cooldown_frames
        self.last_fall_time = 0
        
        print(f"Model loaded: {model_path}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Device: {self.model.device}")
    
    def calculate_fps(self):
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_queue.append(fps)
        return np.mean(self.fps_queue)
    
    def draw_info_panel(self, frame, fps, detections, cooldown_active=False):
        """Draw information panel on frame"""
        height, width = frame.shape[:2]
        panel_height = 150
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        # Display information
        y_offset = 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                    font, 0.7, color, 2)
        
        y_offset += 30
        cv2.putText(frame, f"Detections: {len(detections)}", (10, y_offset), 
                    font, 0.7, color, 2)
        
        y_offset += 30
        cv2.putText(frame, f"Total Falls: {self.fall_count}", (10, y_offset), 
                    font, 0.7, (0, 255, 255), 2)
        
        # Show cooldown status
        if cooldown_active:
            y_offset += 30
            remaining = self.alert_cooldown
            cv2.putText(frame, f"Cooldown: {remaining} frames", (10, y_offset), 
                        font, 0.6, (100, 200, 255), 2)
        
        # Instructions
        y_offset += 30
        cv2.putText(frame, "Press 'Q' to quit | 'R' to reset stats", 
                    (10, y_offset), font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_fall_alert(self, frame):
        """Draw fall detection alert"""
        height, width = frame.shape[:2]
        
        # Blinking alert
        if int(time.time() * 3) % 2:  # Blink 3 times per second
            # Draw red border
            thickness = 15
            color = (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (width, height), color, thickness)
            
            # Alert text
            alert_text = "!!! FALL DETECTED !!!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            text_thickness = 4
            
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(
                alert_text, font, font_scale, text_thickness)
            
            # Position at top center
            x = (width - text_width) // 2
            y = 60
            
            # Background rectangle
            padding = 10
            cv2.rectangle(frame, 
                         (x - padding, y - text_height - padding),
                         (x + text_width + padding, y + padding),
                         (0, 0, 255), -1)
            
            # Text
            cv2.putText(frame, alert_text, (x, y), font, 
                       font_scale, (255, 255, 255), text_thickness)
        
        return frame
    
    def process_detections(self, results):
        """Process detection results and check for falls"""
        detections = []
        fall_detected = False
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names.get(class_id, str(class_id))
                
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
                
                # Check if this is a fall detection
                # Adjust class name based on your model's output
                if 'fall' in class_name.lower():
                    fall_detected = True
        
        return detections, fall_detected
    
    def run(self, save_output=None):
        """
        Run real-time fall detection
        
        Args:
            save_output (str): Optional path to save video output
        """
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nCamera resolution: {width}x{height}")
        print(f"Starting real-time detection...")
        print(f"Press 'Q' to quit, 'R' to reset statistics\n")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, 20.0, (width, height))
            print(f"Recording to: {save_output}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Process detections
                detections, fall_detected = self.process_detections(results)
                
                # Update statistics and handle fall detection
                self.total_detections += len(detections)
                
                # Cooldown management
                if self.alert_cooldown > 0:
                    self.alert_cooldown -= 1
                
                # Only register new fall if cooldown is over
                if fall_detected and self.alert_cooldown == 0:
                    self.fall_count += 1
                    self.alert_cooldown = self.cooldown_frames  # Start cooldown
                    self.alert_active = True
                    self.last_fall_time = time.time()
                    print(f"\n*** FALL DETECTED #{self.fall_count} at {time.strftime('%H:%M:%S')} ***")
                
                # Keep alert visible for first 30 frames of cooldown
                if self.alert_cooldown > (self.cooldown_frames - 30):
                    self.alert_active = True
                else:
                    self.alert_active = False
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw information panel
                cooldown_active = self.alert_cooldown > 0
                annotated_frame = self.draw_info_panel(
                    annotated_frame, fps, detections, cooldown_active)
                
                # Draw fall alert if active (only for first 1 second)
                if self.alert_active:
                    annotated_frame = self.draw_fall_alert(annotated_frame)
                
                # Display frame
                cv2.imshow('Real-time Fall Detection', annotated_frame)
                
                # Save frame if recording
                if writer:
                    writer.write(annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nStopping detection...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.fall_count = 0
                    self.total_detections = 0
                    print("\nStatistics reset")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\nSession Summary:")
            print(f"  Total falls detected: {self.fall_count}")
            print(f"  Total detections: {self.total_detections}")
            if save_output:
                print(f"  Video saved to: {save_output}")


def main():
    """Main function to run real-time fall detection"""

    import os

    # ============ CONFIGURATION ============
    # Use the model file from this workspace by default
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
    CAMERA_INDEX = 0  # 0 for default webcam, 1, 2, etc. for other cameras
    CONFIDENCE_THRESHOLD = 0.9
    SAVE_OUTPUT = os.path.join(BASE_DIR, "output_video.mp4")

    # Cooldown frames between fall detections (90 frames = ~3 seconds at 30fps)
    # Increase this if you get too many repeated detections
    COOLDOWN_FRAMES = 90
    # ======================================

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the script")
        return
    
    # Create detector
    detector = RealtimeFallDetector(
        model_path=MODEL_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        camera_index=CAMERA_INDEX,
        cooldown_frames=COOLDOWN_FRAMES
    )
    
    # Run real-time detection
    detector.run(save_output=SAVE_OUTPUT)


if __name__ == "__main__":
    main()