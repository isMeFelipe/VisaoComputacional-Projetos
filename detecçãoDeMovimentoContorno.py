import cv2
import os
import shutil
import numpy as np

# Inicializar a captura de vídeo (substitua pelo caminho correto do vídeo)
video = cv2.VideoCapture("videoVisao.mp4")

# Verificar se o vídeo foi aberto corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Criar pasta para salvar os frames (sobrescreve se já existir)
output_folder = "frames_capturados"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Remove a pasta e seu conteúdo
os.makedirs(output_folder)

# Obter as propriedades do vídeo original (largura, altura, FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Variável para armazenar o frame anterior
previous_frame = None

# Imagem de rastro (inicializada como zero)
trail_frame = None

# Taxa de esmaecimento do rastro
decay_rate = 0.05

# Contador de frames
frame_count = 0

# Loop principal para processar os frames do vídeo
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Converter o frame atual para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar desfoque gaussiano
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Se não houver frame anterior, continuar
    if previous_frame is None:
        previous_frame = gray_frame
        trail_frame = np.zeros_like(frame, dtype=np.float32)  # float32 para o rastro
        continue

    # Calcular a diferença entre o frame atual e o anterior
    frame_diff = cv2.absdiff(previous_frame, gray_frame)

    # Aplicar threshold
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    # Aplicar operações morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Criar máscara de movimento (pixels em vermelho)
    mask = np.zeros_like(frame, dtype=np.uint8)
    mask[thresh > 0] = [0, 0, 255]

    # Esmaecer o rastro
    trail_frame = cv2.multiply(trail_frame, 1 - decay_rate)

    # Combinar frame original com a máscara vermelha
    frame_with_trail = cv2.add(frame, mask)

    # Sobrepor o rastro
    combined_frame = cv2.addWeighted(frame_with_trail, 0.7, trail_frame.astype(np.uint8), 0.3, 0)

    # Mostrar a imagem com rastro
    cv2.imshow('Movimento Detectado com Rastro em Vermelho', combined_frame)

    # Atualizar o frame anterior
    previous_frame = gray_frame

    # **Salvar o frame na pasta**
    frame_filename = f"{output_folder}/frame_{frame_count:04d}.jpg"
    cv2.imwrite(frame_filename, combined_frame)


    # Incrementar o contador de frames
    frame_count += 1

    # Sair se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video.release()
cv2.destroyAllWindows()

print(f"Todos os frames foram salvos em '{output_folder}'.")
