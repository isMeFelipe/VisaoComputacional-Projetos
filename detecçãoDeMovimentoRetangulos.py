import cv2
import numpy as np

# Inicializar a captura de vídeo (0 para webcam, ou substitua por um caminho de arquivo de vídeo)
video = cv2.VideoCapture("videoVisao.mp4")

# Inicializar a variável para armazenar o frame anterior
previous_frame = None

# Criar uma "imagem de rastro" com o mesmo tamanho dos frames de vídeo, inicializada com zeros (preto)
trail_frame = None

# Taxa de esmaecimento do rastro (ajuste esse valor para mais ou menos esmaecimento)
decay_rate = 0.03  # Diminuir 5% da intensidade a cada frame

# Loop principal para processar os frames do vídeo
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Converter o frame atual para escala de cinza
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar desfoque gaussiano para reduzir o ruído e suavizar a imagem
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Se o frame anterior ainda não existe, armazenar o primeiro frame e continuar
    if previous_frame is None:
        previous_frame = gray_frame
        # Inicializar o frame de rastro com zeros (mesmo tamanho do frame original)
        trail_frame = np.zeros_like(frame, dtype=np.float32)  # float32 para acumulação gradual
        continue

    # Calcular a diferença absoluta entre o frame atual e o anterior
    frame_diff = cv2.absdiff(previous_frame, gray_frame)

    # Aplicar limiar (threshold) para obter uma imagem binária
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)


    # Detectar contornos nas regiões de movimento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar retângulos ao redor dos contornos detectados e adicionar ao frame de rastro
    for contour in contours:
        if cv2.contourArea(contour) < 1800:
            # Ignorar pequenos contornos (áreas de menos de 500 pixels)
            continue

        # Obter as coordenadas do retângulo que envolve o contorno
        (x, y, w, h) = cv2.boundingRect(contour)
        # Desenhar o retângulo no frame de rastro (usando float para manter a precisão)
        cv2.rectangle(trail_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Gradualmente diminuir a intensidade do rastro (decay)
    trail_frame = cv2.multiply(trail_frame, 1 - decay_rate)

    # Converter o trail_frame de float32 para uint8 para sobrepor na imagem original
    trail_frame_uint8 = np.clip(trail_frame, 0, 255).astype(np.uint8)

    # Sobrepor o frame de rastro no frame atual
    combined_frame = cv2.addWeighted(frame, 0.7, trail_frame_uint8, 0.3, 0)

    # Mostrar a imagem com os contornos de movimento detectado e o rastro
    cv2.imshow('Movimento Detectado com Rastro', combined_frame)

    # Atualizar o frame anterior para o próximo loop
    previous_frame = gray_frame

    # Sair se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
video.release()
cv2.destroyAllWindows()
