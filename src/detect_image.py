from config import BEST_MODEL_PATH, DETECTION_IMAGES_DIR
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# טען את המודל
model = YOLO(BEST_MODEL_PATH)

# טען את התמונה
image_path = "../data/images/img165272.jpg"
img = cv2.imread(image_path)


results = model(img, conf=0.6)

# קבל את התיבות
boxes = results[0].boxes

# אם אין תיבות בכלל, פשוט תדפיסי הודעה
if boxes is None or len(boxes) == 0:
    print("לא זוהה שום אובייקט עם רמת ביטחון מעל 0.6")
else:
    for box in boxes:
        confidence = box.conf[0].item()
        if confidence >= 0.6:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls = int(box.cls[0].item())
            label = results[0].names[cls]

            print(f"תיבת זיהוי: ({x1}, {y1}), ({x2}, {y2})")
            print(f"רמת ביטחון: {confidence:.2f}")
            print(f"קטגוריה: {label}")

            # צייר מלבן על התמונה
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # שמירת התמונה בתיקיית הזיהויים
    output_filename = os.path.basename(image_path).replace(" ", "_")
    output_path = os.path.join(DETECTION_IMAGES_DIR, f"detected_{output_filename}")
    cv2.imwrite(output_path, img)
    print(f"התמונה נשמרה בנתיב: {output_path}")

    # הצגת התמונה
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("תמונה אחרי זיהוי")
    plt.axis('off')
    plt.show()
