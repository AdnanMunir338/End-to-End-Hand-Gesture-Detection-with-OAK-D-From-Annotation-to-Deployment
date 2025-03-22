
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('hand_recg_m.pt')

# Export the model to ONNX format
# model.export(format='onnx')

# Set up the RealSense D455 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Set the depth scale
depth_scale = 0.0010000000474974513

# Create spatial, temporal, and disparity filters
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
disparity_transform = rs.disparity_transform(True)  # Enable disparity transform

# Main loop
try:
    while True:
        # Get the latest frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert the frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply filters to depth image
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_transform.process(depth_frame)  # Convert to disparity map
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the depth image to meters
        depth_image = depth_image * depth_scale

        # Detect objects using YOLOv8
        results = model(color_image)

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy()

                if confidence < 0.5:
                    continue  # Skip detections with low confidence

                # Calculate the distance to the object
                object_depth = np.median(depth_image[y1:y2, x1:x2])

                # Get class name from YOLOv8 model
                class_name = model.names[int(class_id)]

                # Combine class name and distance in the label
                label = f"{class_name} {object_depth:.2f}m"

                # Draw bounding box
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (252, 119, 30), 2)

               
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(color_image, (x1, y1 - 25), (x1 + w, y1), (252, 119, 30), -1)
                cv2.putText(color_image, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

                # Print to console
                print(f"{class_name}: {object_depth:.2f}m")

        # Show the color image
        cv2.imshow("Color Image", color_image)

        # Show the disparity map
        disparity_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("Disparity Map", depth_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()