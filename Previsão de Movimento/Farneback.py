import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans

def process_video(video_path, output_dir="processed_frames"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Erro ao abrir o vídeo.")
        return
    
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Parâmetros
    movement_threshold = 2  # Limite para considerar movimento
    num_clusters = 4  # Número de objetos móveis distintos 
    background_color = [50, 50, 50]  # Cor fixa para o fundo
    
    frame_count = 0 
    
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(    # Calcular fluxo óptico usando Farneback
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])         # Calcular magnitude e ângulo do fluxo

        
        mask = np.zeros_like(frame2)
        mask[:] = background_color  # Cor fixa para o fundo

        # Identificar objetos em movimento
        moving_mask = magnitude >= movement_threshold
        if np.any(moving_mask):
            # Normalizar ângulo e magnitude para clustering
            features = np.column_stack((
                angle[moving_mask].flatten(),
                magnitude[moving_mask].flatten()
            ))

            # Aplicar clustering (K-means) somente se houver pixels suficientes
            if len(features) >= num_clusters:
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                labels = kmeans.fit_predict(features)

                # Definir cores para clusters
                colors = np.random.randint(50, 255, (num_clusters, 3), dtype=np.uint8)
                
                # Obter índices reais dos pixels em movimento
                moving_indices = np.array(np.where(moving_mask)).T

                # Atribuir cores aos clusters corretamente
                for idx, (y, x) in enumerate(moving_indices):
                    mask[y, x] = colors[labels[idx]]

        # Combinar o frame original com a máscara
        combined = cv2.addWeighted(frame2, 0.5, mask, 0.5, 0)

        # Salvar o frame processado
        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(output_path, combined)
        
        cv2.imshow("Segmentação de Movimento", combined)

        # Atualizar o frame anterior
        prev_gray = gray
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == 27:  # Pressionar 'Esc' para sair
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames processados salvos na pasta: {output_dir}")

process_video("videoVisao.mp4")
