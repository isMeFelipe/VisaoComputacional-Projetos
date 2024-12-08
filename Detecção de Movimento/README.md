Requisitos:

pip install opencv-python


# Opções de execução
python .\detecçãoDeMovimentoContorno.py (Relatório)

python .\detecçãoDeMovimentoExpressiva.py (Aplicação de rastro com mais expressividade, apenas por curiosidade)
    --> Adição dessa linha 
    --> trail_frame = cv2.add(trail_frame, mask.astype(np.float32))