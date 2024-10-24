import cv2
cap = cv2.VideoCapture('11.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
start_time = 0
end_time = 30
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)
# Устанавливаем начальный кадр
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_11.mp4', fourcc, fps, (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
))
while cap.isOpened():
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if current_frame >= end_frame:
        break

    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
cap.release()
out.release()
