import os

# נתיב לתיקיית הבסיס של הפרויקט (התיקייה שבה נמצא config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# נתיבי המודלים
MODEL_PATH_1 = os.path.join(BASE_DIR, "models", "best_exp1.pt")
MODEL_PATH_2 = os.path.join(BASE_DIR, "models", "best_exp2.pt")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# נתיב בסיסי לשמירת כל תוצרי הפרויקט
BASE_OUTPUT_DIR = os.path.join(BASE_DIR, "project_outputs")

# תתי-תיקיות לתוצרי הרצה
DETECTION_IMAGES_DIR = os.path.join(BASE_OUTPUT_DIR, "detections")
DETECTION_VIDEOS_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
EXTRACTED_FRAMES_DIR = os.path.join(BASE_OUTPUT_DIR, "frames")

# כתובת הזרמת הווידאו מהמצלמה
CAMERA_URL = 'http://192.168.137.89:81/stream'

# כתובת לשליחת התראה ל-ESP32
ESP32_ALERT_URL = 'http://192.168.137.109/alert'

BASE_PATH = os.path.join(os.environ["USERPROFILE"], "Downloads", "Viper UAV.v10i.yolov8")
