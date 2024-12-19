import cv2
import numpy as np

class CSRTTracker:
    def __init__(self, roi):
        self.roi = roi
        self.prev_frame = None
        self.current_frame = None
        self.points = None  # Ключевые точки в ROI
        self.bounding_box_margin = 20  # Добавляем запас к границам ROI

    def initialize_tracks(self, frame):
        self.prev_frame = frame
        self.current_frame = frame
        self.__recalculate_features(frame)

    def __recalculate_features(self, frame):
        # Обновляем ключевые точки в границах ROI
        roi_frame = frame[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
        self.points = cv2.goodFeaturesToTrack(roi_frame, maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=7)

        # Смещение координат точек в глобальные (ROI к полному кадру)
        if self.points is not None:
            self.points[:, 0, 0] += self.roi[0]
            self.points[:, 0, 1] += self.roi[1]

    def update(self, frame):
        self.current_frame = frame

        if self.points is None or len(self.points) == 0:
            self.__recalculate_features(frame)  # Если потеряли объект, пересчитываем точки
            return self.roi

        # Оптический поток
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, self.current_frame, self.points, None,
            winSize=(15, 15), maxLevel=3, criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT, 10, 0.03)
        )

        if p1 is None or len(p1[st == 1]) == 0:
            self.__recalculate_features(frame)  # Если нет хороших точек, пересчитываем точки
            return self.roi

        # Выбираем "хорошие" точки
        good_new = p1[st == 1].reshape(-1, 2)
        good_old = self.points[st == 1].reshape(-1, 2)

        # Расчет среднего смещения
        dx = np.mean(good_new[:, 0] - good_old[:, 0])
        dy = np.mean(good_new[:, 1] - good_old[:, 1])

        # Обновляем ROI с учетом ограничений кадра
        roi_new_x = max(0, min(self.roi[0] + int(dx), frame.shape[1] - self.roi[2]))
        roi_new_y = max(0, min(self.roi[1] + int(dy), frame.shape[0] - self.roi[3]))
        self.roi = (roi_new_x, roi_new_y, self.roi[2], self.roi[3])

        # Обновляем ключевые точки в новых границах ROI
        self.__recalculate_features(frame)

        # Обновляем прошлый кадр
        self.prev_frame = frame

        return self.roi


# Инициализация видео источника
video_source = "video4.mp4"
cap = cv2.VideoCapture(video_source)

# Прочитать первый кадр
ret, frame = cap.read()
if not ret:
    print("Не удалось прочитать кадр из видео.")
    cap.release()
    exit()

# Уменьшение масштаба первого кадра для лучшей производительности
scale = 0.5  # коэффициент масштаба (уменьшаем до 50% от оригинала)
frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

# Выбор ROI для трекинга
roi = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

# Инициализация трекера
tracker = CSRTTracker(roi)
tracker.initialize_tracks(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

while True:
    # Прочитать следующий кадр
    ret, frame = cap.read()
    if not ret:
        break

    # Уменьшение масштаба кадра

    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    # Конвертируем кадр в оттенки серого для трекинга
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обновляем положение ROI
    roi = tracker.update(gray_frame)

    # Отображение ROI прямоугольником
    cv2.rectangle(
        frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2
    )

    # Показать кадр с выбранным ROI
    cv2.imshow("Tracking", frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
