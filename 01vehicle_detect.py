import cv2

from ultralytics import YOLO

cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    raise IOError("Please check ypu video file again")

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = [2, 3, 5, 7]

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = model(frame, verbose = False)
    
    for box in results[0].boxes:
        
        class_id = int(box.cls[0])
        
        if class_id not in VEHICLE_CLASSES:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        confidence = float(box.conf[0])
        
        class_name = model.names[class_id]
        
        
        cv2.rectangle(
               frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )
        
        label = f"{class_name} {confidence:.2f}"
        
        cv2.putText(
                frame,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
    