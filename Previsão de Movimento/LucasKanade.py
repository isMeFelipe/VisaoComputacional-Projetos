import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

def processar_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Erro ao abrir o vídeo.')
        return

    frames_dir = './processed_frames'
    if os.path.exists(frames_dir):
        for file in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, file))
    else:
        os.makedirs(frames_dir)

    colors = np.random.randint(0, 255, (10, 3), dtype=np.uint8)     # Lista de cores aleatórias para preencher objetos detectados

    frame_idx = 0

    ret, prev_frame = cap.read()

    if not ret:
        print('Erro ao ler o primeiro frame.')
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.14, minDistance=0)  # Configuração de threshhold

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, winSize=(50, 50)) # Calcular o fluxo óptico usando Lucas-Kanade

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                # Agrupar os pontos com k-means
                kmeans = KMeans(n_clusters=min(5, len(good_new)))
                kmeans.fit(good_new)
                labels = kmeans.labels_

                for i in np.unique(labels):
                    cluster_points = good_new[labels == i].astype(int)

                    mask = np.zeros_like(frame, dtype=np.uint8)

                    color = tuple(np.random.choice(256, 3).tolist())
                    for point in cluster_points:
                        x, y = point
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv2.circle(mask, (x, y), 5, color, -1)

                    frame = cv2.add(frame, mask)

        prev_gray = frame_gray.copy()

        frame_path = os.path.join(frames_dir, f'frame_{frame_idx}.jpg')
        cv2.imwrite(frame_path, frame)

        cv2.imshow('Vídeo Processado', frame)

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27:         # Pressionar 'Esc' para sair do vídeo
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f'Frames segmentados salvos em {frames_dir}')


processar_video('./videoVisao.mp4')
