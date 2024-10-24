import cv2

def get_video_info(filename):

    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return f"Невозможно открыть видеофайл: {filename}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return f"Невозможно определить FPS для {filename}"

    # Получаем количество кадров
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps else 0

    # Получаем кодек
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    # Рассчитываем интенсивность объектов
    intensity = 0
    frame_sample_count = 10  # Количество фреймов для выборки
    step = max(1, int(frame_count) // frame_sample_count)  # Шаг через который производится выборка

    for i in range(0, frame_sample_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensity += cv2.mean(gray_frame)[0]  # Среднее значение яркости
        # канала с  индексом 1 просто нет в черно - белом изображении.

    intensity /= (frame_sample_count)  # Делим на 10 кадров, чтобы интенсивность была меньше

    cap.release()

    return {
        "filename": filename,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "codec": codec_str,
        "intensity": intensity
    }
print(get_video_info('video1.mp4'))
print(get_video_info('video2.mp4'))
print(get_video_info('video3.mp4'))
print(get_video_info('video4.mp4'))
print(get_video_info('video5.mp4'))

