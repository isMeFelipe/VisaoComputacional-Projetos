import cv2
import os
import shutil
import numpy as np

video = cv2.VideoCapture("videoVisao.mp4")

if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Criar pasta para salvar os frames (sobrescreve se já existir)
output_folder = "frames_capturados"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Remove a pasta e seu conteúdo
os.makedirs(output_folder)

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


previous_frame = None

trail_frame = None    # Imagem de rastro (inicializada como zero)

decay_rate = 0.5

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if previous_frame is None:
        previous_frame = gray_frame
        trail_frame = np.zeros_like(frame, dtype=np.float32)
        continue

    frame_diff = cv2.absdiff(previous_frame, gray_frame)

    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    mask = np.zeros_like(frame, dtype=np.uint8)
    mask[thresh > 0] = [0, 0, 255]

    trail_frame = cv2.multiply(trail_frame, 1 - decay_rate)

    frame_with_trail = cv2.add(frame, mask)

    combined_frame = cv2.addWeighted(frame_with_trail, 0.7, trail_frame.astype(np.uint8), 0.3, 0)

    cv2.imshow('Movimento Detectado com Rastro em Vermelho', combined_frame)

    previous_frame = gray_frame

    frame_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
    cv2.imwrite(frame_filename, combined_frame)


    frame_count += 1

    # Sair se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print(f"Todos os frames foram salvos em '{output_folder}'.")
