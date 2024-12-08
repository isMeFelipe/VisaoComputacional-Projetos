import cv2
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans

def horn_schunck_optical_flow(prev_gray, gray, alpha=0.1, num_iter=100):
    """
    Estimativa de fluxo óptico usando o método Horn-Schunck.
    :param prev_gray: Frame anterior em escala de cinza.
    :param gray: Frame atual em escala de cinza.
    :param alpha: Parâmetro de regularização (ajustável).
    :param num_iter: Número de iterações para otimização.
    :return: u, v - Componentes horizontal e vertical do fluxo.
    """
    u = np.zeros_like(gray, dtype=np.float32)
    v = np.zeros_like(gray, dtype=np.float32)

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    It = prev_gray - gray

    for _ in range(num_iter):
        u_avg = cv2.GaussianBlur(u, (5, 5), 1)
        v_avg = cv2.GaussianBlur(v, (5, 5), 1)

        P = Ix * u_avg + Iy * v_avg + It
        D = alpha**2 + Ix**2 + Iy**2

        u = u_avg - Ix * P / D
        v = v_avg - Iy * P / D

    return u, v

def process_video(video_path, output_dir="processed_frames"):
    # Configura a pasta de saída
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret:
        print("Erro ao abrir o vídeo.")
        return

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    movement_threshold = 80
    num_clusters = 4
    background_color = [0, 0, 0]

    frame_count = 0

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calcula fluxo óptico com Horn-Schunck
        u, v = horn_schunck_optical_flow(prev_gray, gray)

        magnitude = np.sqrt(u**2 + v**2)

        mask = np.zeros_like(frame2)
        mask[:] = background_color

        moving_mask = magnitude >= movement_threshold
        if np.any(moving_mask):
            features = np.column_stack((u[moving_mask].flatten(), v[moving_mask].flatten()))

            if len(features) >= num_clusters:
                kmeans = KMeans(n_clusters=num_clusters, random_state=0)
                labels = kmeans.fit_predict(features)

                colors = np.random.randint(50, 255, (num_clusters, 3), dtype=np.uint8)

                moving_indices = np.array(np.where(moving_mask)).T

                for idx, (y, x) in enumerate(moving_indices):
                    mask[y, x] = colors[labels[idx]]

        combined = cv2.addWeighted(frame2, 0.5, mask, 0.5, 0)

        output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(output_path, combined)

        cv2.imshow("Segmentação de Movimento", combined)

        prev_gray = gray
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames processados salvos na pasta: {output_dir}")

process_video("videoVisao.mp4")
