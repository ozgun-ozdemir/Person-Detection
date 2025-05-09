from ultralytics import YOLO
import cv2

model = YOLO('yolo11n-pose.pt')
model.to("mps") # MPS (Apple Silicon)|cuda (NVIDIA GPU)

video_path = 'media1.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=0.4, # confidence threshold
        iou=0.3, # IoU threshold
        classes=[0] # only person class
    )
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else []
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(ids[i]) if len(ids) > i else -1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
            if len(keypoints) > i:
                person_kpts = keypoints[i] 
                for kp in person_kpts:
                    cx, cy = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()