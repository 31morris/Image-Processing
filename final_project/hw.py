#!/usr/bin/python3
from ultralytics import YOLO
import cv2
import os

# 設定影片來源路徑和儲存目錄
source = 'new_input3/person_elephants_V3.mp4'
save_dir_p = './output_frames'  # 儲存影格的資料夾
save_dir_v = './output_video'  # 儲存輸出影片的資料夾
os.makedirs(save_dir_p, exist_ok=True)
os.makedirs(save_dir_v, exist_ok=True)

# 載入 YOLOv10 模型
model = YOLO('yolov10x.pt')

# 開啟輸入影片
cap = cv2.VideoCapture(source)

# 取得影片屬性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 影片寬度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 影片高度
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 每秒影格數
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 指定影片的編碼格式
out = cv2.VideoWriter(os.path.join(save_dir_v, 'output_video.mp4'), fourcc, fps, (frame_width, frame_height))

# 進行人與大象的檢測
results_person = model.predict(source=source, classes=[0], imgsz=1440, conf=0.1, iou=0.5)  # 檢測人類
results_elephant = model.predict(source=source, classes=[20], imgsz=1280, conf=0.5, iou=0.85)  # 檢測大象

# 遍歷影片影格並合併檢測結果
for i, (result_person, result_elephant) in enumerate(zip(results_person, results_elephant)):
    # 取得原始影像
    orig_image = result_person.orig_img.copy()

    # 合併人和大象的檢測框
    combined_boxes = []
    combined_classes = []
    combined_confs = []

    if result_person.boxes is not None:
        combined_boxes.extend(result_person.boxes.xyxy.cpu().numpy())  # 人類檢測框
        combined_classes.extend(result_person.boxes.cls.cpu().numpy())  # 人類類別
        combined_confs.extend(result_person.boxes.conf.cpu().numpy())  # 人類置信度
    
    if result_elephant.boxes is not None:
        combined_boxes.extend(result_elephant.boxes.xyxy.cpu().numpy())  # 大象檢測框
        combined_classes.extend(result_elephant.boxes.cls.cpu().numpy())  # 大象類別
        combined_confs.extend(result_elephant.boxes.conf.cpu().numpy())  # 大象置信度

    # 繪製檢測框
    for box, cls, conf in zip(combined_boxes, combined_classes, combined_confs):
        x1, y1, x2, y2 = map(int, box)  # 取得檢測框座標
        class_name = result_person.names[int(cls)]  # 取得類別名稱

        if class_name == "person":
            color = (0, 0, 255)  # 紅色代表人類
        elif class_name == "elephant":
            color = (0, 255, 0)  # 綠色代表大象
        else:
            color = (255, 255, 255)  # 白色代表其他

        # 繪製檢測框及標籤
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), color, 4)
        label = f"{class_name} {conf:.2f}"  # 顯示類別與置信度
        cv2.putText(orig_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # 計算人類與大象的數量
    class_human_count = sum(1 for cls in combined_classes if cls == 0)
    class_elephant_count = sum(1 for cls in combined_classes if cls == 20)

    # 在影格上顯示學號與檢測數量
    red_color = (0, 0, 255)  # 紅色
    cv2.putText(orig_image, f"313512072", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2)
    cv2.putText(orig_image, f"person: {class_human_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2)
    cv2.putText(orig_image, f"elephant: {class_elephant_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, red_color, 2)

    # 儲存處理後的影格
    frame_file_path = os.path.join(save_dir_p, f'frame_{i}.jpg')
    cv2.imwrite(frame_file_path, orig_image)
    out.write(orig_image)

# 釋放資源
cap.release()
out.release()
cv2.destroyAllWindows()
