import cv2
import os


def FrameCapture(path, seconds_per_frame=1):
    # פתיחת קובץ הווידאו
    vidObj = cv2.VideoCapture(path)

    # קבלת מספר פריימים לשנייה (FPS)
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    print(f"FPS של הסרטון: {fps}")

    # חישוב כל כמה פריימים לשמור תמונה
    frames_interval = int(fps * seconds_per_frame)
    print(f"שומרים כל {frames_interval} פריימים")

    # יצירת תיקיית הפלט אם היא לא קיימת
    output_folder = 'frames'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    saved_count = 0
    success, image = vidObj.read()

    while success:
        if count % frames_interval == 0:
            if image is not None:
                filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, image)
                saved_count += 1

        count += 1
        success, image = vidObj.read()

    vidObj.release()
    print(f"נשמרו {saved_count} תמונות בתיקייה '{output_folder}'")

if __name__ == '__main__':
    FrameCapture(r"C:\אימון מודל\20250402-2005-17.2223385.mp4", seconds_per_frame=1)
