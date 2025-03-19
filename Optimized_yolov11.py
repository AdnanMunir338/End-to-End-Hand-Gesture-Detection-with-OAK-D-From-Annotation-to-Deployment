
import cv2
import depthai as dai
import numpy as np
import time

# ---------------------------- CONFIGURATION ---------------------------- #
FRAME_WIDTH, FRAME_HEIGHT = 640, 640  # Match NN input size
NN_WIDTH, NN_HEIGHT = 640, 640  # YOLOv8 input size
CONFIDENCE_THRESHOLD = 0.4  
IOU_THRESHOLD = 0.7  

# COCO Dataset Labels
LABELS = ["five", "four", "one", "three", "two"]

# ---------------------------- CREATE PIPELINE ---------------------------- #
pipeline = dai.Pipeline()

# Define sources and outputs
cam_rgb = pipeline.create(dai.node.ColorCamera)
spatial_detection_network = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("rgb")
xout_nn.setStreamName("detections")
xout_depth.setStreamName("depth")

# ---------------------------- CAMERA SETTINGS ---------------------------- #
cam_rgb.setPreviewSize(FRAME_WIDTH, FRAME_HEIGHT)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stereo depth settings
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align depth with RGB camera
stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
stereo.setLeftRightCheck(True)  # Required for depth alignment
stereo.setSubpixel(False)  # Disable for better performance

# ---------------------------- YOLOv8 MODEL SETTINGS ---------------------------- #
# spatial_detection_network.setBlobPath(r"C:\Users\adnan\Downloads\OAK-Object-Detection-with-Depth\result\yolov8n_openvino_2022.1_6shave.blob")
spatial_detection_network.setBlobPath(r"C:\Users\adnan\Downloads\OAK-Object-Detection-with-Depth\hand_recog_onnx\yolo11n\yolo11_hand_recg_n_openvino_2022.1_6shave.blob")
spatial_detection_network.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
spatial_detection_network.input.setBlocking(False)
spatial_detection_network.setBoundingBoxScaleFactor(0.3)
spatial_detection_network.setDepthLowerThreshold(100)
spatial_detection_network.setDepthUpperThreshold(5000)

# YOLO-specific parameters
spatial_detection_network.setNumClasses(len(LABELS))
spatial_detection_network.setCoordinateSize(4)
spatial_detection_network.setAnchors([
    10, 13, 16, 30, 33, 23,
    30, 61, 62, 45, 59, 119,
    116, 90, 156, 198, 373, 326
])
spatial_detection_network.setAnchorMasks({
    "side26": [0, 1, 2],
    "side13": [3, 4, 5]
})
spatial_detection_network.setIouThreshold(IOU_THRESHOLD)

# ---------------------------- LINKING NODES ---------------------------- #
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

cam_rgb.preview.link(spatial_detection_network.input)
spatial_detection_network.passthrough.link(xout_rgb.input)
spatial_detection_network.out.link(xout_nn.input)

stereo.depth.link(spatial_detection_network.inputDepth)
stereo.depth.link(xout_depth.input)

# ---------------------------- RUN PIPELINE ---------------------------- #
with dai.Device(pipeline) as device:
    # Output queues
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    q_det = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []
    start_time = time.monotonic()
    counter = 0
    fps = 0

    while True:
        in_rgb = q_rgb.get()
        in_det = q_det.get()
        in_depth = q_depth.get()

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        # Normalize depth frame for visualization
        depth_frame_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_frame_colored = cv2.applyColorMap(depth_frame_normalized, cv2.COLORMAP_JET)

        detections = in_det.detections

        # Calculate FPS
        counter += 1
        current_time = time.monotonic()
        if (current_time - start_time) > 1:
            fps = counter / (current_time - start_time)
            counter = 0
            start_time = current_time

        # Draw detections
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * FRAME_WIDTH)
            y1 = int(detection.ymin * FRAME_HEIGHT)
            x2 = int(detection.xmax * FRAME_WIDTH)
            y2 = int(detection.ymax * FRAME_HEIGHT)

            label = LABELS[detection.label] if detection.label < len(LABELS) else "Unknown"
            confidence = detection.confidence
            depth_mm = detection.spatialCoordinates.z  # Get depth in mm

            print(f"Object: {label}, Confidence: {confidence:.2f}, Depth: {depth_mm:.2f} mm")

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Depth: {depth_mm:.2f} mm", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show output
        cv2.imshow("YOLO Object Detection", frame)
        cv2.imshow("Depth Map", depth_frame_colored)

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


