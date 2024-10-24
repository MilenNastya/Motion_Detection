import cv2
import time

video = int(input("Video: "))

if video == 1:
    cap = cv2.VideoCapture('video1.MP4')
elif video == 2:
    cap = cv2.VideoCapture('video2.MP4')
elif video == 3:
    cap = cv2.VideoCapture('video3.MP4')
elif video == 4:
    cap = cv2.VideoCapture('video4.MP4')
elif video == 5:
    cap = cv2.VideoCapture('video5.MP4')
else:
    print("fault!")

if not cap.isOpened():
    print("Ошибка: не удалось открыть видеофайл.")
    exit()

choose = int(input("Tracker: "))

if choose == 1:
    tracker = cv2.TrackerCSRT.create()
elif choose == 2:
    tracker = cv2.legacy.TrackerKCF.create()
else:
    tracker = cv2.legacy.TrackerTLD.create()

ret, frame = cap.read()
if not ret:
    print("Ошибка: не удалось прочитать первый кадр.")
    exit()

# Определение размеров кадра для уменьшенной версии
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(orig_width * 0.5)
height = int(orig_height * 0.5)

# Изменение размера первого кадра и увеличение контраста
frame_resized = cv2.resize(frame, (width, height))
bbox_resized = cv2.selectROI("Выберите область для отслеживания", frame_resized, False)
cv2.destroyWindow("Выберите область для отслеживания")
bbox = (bbox_resized[0] * 2, bbox_resized[1] * 2, bbox_resized[2] * 2, bbox_resized[3] * 2)

tracker.init(frame, bbox)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
if choose == 1:
    if video == 1:
        out = cv2.VideoWriter('video_csrt_1.mp4', fourcc, 20.0, (width, height))
    elif video == 2:
        out = cv2.VideoWriter('video_csrt_2.mp4', fourcc, 20.0, (width, height))
    elif video == 3:
        out = cv2.VideoWriter('video_csrt_3.mp4', fourcc, 20.0, (width, height))
    elif video == 4:
        out = cv2.VideoWriter('video_csrt_4.mp4', fourcc, 20.0, (width, height))
    elif video == 5:
        out = cv2.VideoWriter('video_csrt_5.mp4', fourcc, 20.0, (width, height))
elif choose == 2:
    if video == 1:
        out = cv2.VideoWriter('video_kcf_1.mp4', fourcc, 20.0, (width, height))
    elif video == 2:
        out = cv2.VideoWriter('video_kcf_2.mp4', fourcc, 20.0, (width, height))
    elif video == 3:
        out = cv2.VideoWriter('video_kcf_3.mp4', fourcc, 20.0, (width, height))
    elif video == 4:
        out = cv2.VideoWriter('video_kcf_4.mp4', fourcc, 20.0, (width, height))
    elif video == 5:
        out = cv2.VideoWriter('video_kcf_5.mp4', fourcc, 20.0, (width, height))
else:
    if video == 1:
        out = cv2.VideoWriter('video_tld_1.mp4', fourcc, 20.0, (width, height))
    elif video == 2:
        out = cv2.VideoWriter('video_tld_2.mp4', fourcc, 20.0, (width, height))
    elif video == 3:
        out = cv2.VideoWriter('video_tld_3.mp4', fourcc, 20.0, (width, height))
    elif video == 4:
        out = cv2.VideoWriter('video_tld_4.mp4', fourcc, 20.0, (width, height))
    elif video == 5:
        out = cv2.VideoWriter('video_tld_5.mp4', fourcc, 20.0, (width, height))


start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Конец видео или ошибка чтения.")
        break
    success, bbox = tracker.update(frame)
    frame_resized = cv2.resize(frame, (width, height))

    if success:
        # Масштабирование координат обратно для уменьшенного кадра
        x, y, w, h = [int(v / 2) for v in bbox]

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame_resized, "Tracking failed!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    out.write(frame_resized)  # Запись уменьшенного кадра
    cv2.imshow('Tracking', frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
if frame_count != 0:
    print(f"Время работы метода: {end_time - start_time:.5f} секунд")
    print(f"Скорость обработки: {frame_count / (end_time - start_time):.2f} кадров/секунду")
    print(f"Частота потери изображения: {1 / ((end_time - start_time) / cap.get(cv2.CAP_PROP_POS_FRAMES)):.0f} кадров/секунду")

cap.release()
out.release()
cv2.destroyAllWindows()
