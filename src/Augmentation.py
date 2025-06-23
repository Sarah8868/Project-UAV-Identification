import os
import cv2
from tqdm import tqdm
import albumentations as A
from config import BASE_PATH




# קריאת תגיות YOLO
def read_yolo_labels(label_path):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                cls, cx, cy, w, h = map(float, parts)
                boxes.append([cls, cx, cy, w, h])
    return boxes

# שמירת תגיות YOLO
def write_yolo_labels(label_path, boxes):
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(" ".join([str(x) for x in box]) + "\n")

# המרה ל־Albumentations
def yolo_to_alb(box, w, h):
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]

# חזרה ל־YOLO
def alb_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]

# Augmentation
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomFog(p=0.2),
    A.RandomShadow(p=0.3),
    A.RandomRain(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# עיבוד תיקייה אחת
def augment_folder(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(src_images_dir), desc=f"Processing {src_images_dir}"):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(src_images_dir, img_name)
        label_path = os.path.join(src_labels_dir, os.path.splitext(img_name)[0] + ".txt")

        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        yolo_boxes = read_yolo_labels(label_path)

        alb_boxes = [yolo_to_alb(box[1:], w, h) for box in yolo_boxes]
        class_labels = [int(box[0]) for box in yolo_boxes]

        transformed = transform(image=image, bboxes=alb_boxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_labels = transformed['class_labels']

        new_img_name = f"aug_{img_name}"
        new_label_name = f"aug_{os.path.splitext(img_name)[0]}.txt"

        cv2.imwrite(os.path.join(dst_images_dir, new_img_name), transformed_image)

        new_boxes = []
        for box, cls in zip(transformed_bboxes, transformed_labels):
            yolo_box = alb_to_yolo(box, w, h)
            new_boxes.append([cls] + yolo_box)

        write_yolo_labels(os.path.join(dst_labels_dir, new_label_name), new_boxes)


for split in ['train', 'valid', 'test']:
    src_images = os.path.join(BASE_PATH, split, "images")
    src_labels = os.path.join(BASE_PATH, split, "labels")
    dst_images = os.path.join(BASE_PATH, f"{split}_aug", "images")
    dst_labels = os.path.join(BASE_PATH, f"{split}_aug", "labels")

    augment_folder(src_images, src_labels, dst_images, dst_labels)
