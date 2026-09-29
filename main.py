from ultralytics import YOLO
import cv2
import os

# --------------------------------
# 1. Video path
# --------------------------------
video_path = "Videos/car_obstacle_detection_test.mp4"

if not os.path.exists(video_path):
    print("Video file not found!")
    exit()

print("Video found!")

# --------------------------------
# 2. Load YOLO model
# --------------------------------
print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded!")

# --------------------------------
# 3. Open video
# --------------------------------
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video!")
    exit()

print("Video opened successfully!")

# --------------------------------
# 4. Process video
# --------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    # Draw detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Get coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Object name
            object_name = model.names[class_id]

            # Only show reasonably confident detections
            if confidence > 0.40:

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                label = f"{object_name} {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # --------------------------------
    # Display
    # --------------------------------
    cv2.imshow(
        "YOLO - Car Obstacle Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------
# 5. Release
# --------------------------------
cap.release()
cv2.destroyAllWindows()

print("Detection completed.")
