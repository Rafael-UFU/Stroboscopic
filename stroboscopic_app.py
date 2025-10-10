import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

# --- Funções de Lógica (do nosso script anterior) ---

def obter_centro(bbox):
    x, y, w, h = [int(v) for v in bbox]
    return (x + w / 2, y + h / 2)

def processar_video(video_bytes, bbox_coords):
    """
    Função principal que processa o vídeo e retorna a imagem e os dados.
    Recebe os bytes do vídeo e as coordenadas da BBox.
    """
    # Salva os bytes do vídeo em um arquivo temporário para o OpenCV ler
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    captura = cv2.VideoCapture(video_path)
    if not captura.isOpened():
        st.error("Erro ao abrir o arquivo de vídeo.")
        return None, None

    sucesso, frame_inicial = captura.read()
    if not sucesso:
        st.error("Não foi possível ler o primeiro frame do vídeo.")
        return None, None
    
    # Inicializa o tracker com as coordenadas recebidas da interface
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_inicial, bbox_coords)

    imagem_estroboscopica = frame_inicial.copy()
    
    # Lógica de dados CSV
    dados_trajetoria = []
    centro_origem = obter_centro(bbox_coords)
    dados_trajetoria.append([0, 0.0, 0.0, centro_origem[0], centro_origem[1]])
    
    # Lógica de "carimbo" adaptativo
    fator_distancia = 0.8 # Pode ser ajustado ou virar uma opção na UI
    largura_objeto = bbox_coords[2]
    distancia_minima_para_carimbar = largura_objeto * fator_distancia
    posicao_ultimo_carimbo = centro_origem

    contador_frames_total = 0
    while True:
        sucesso, frame_atual = captura.read()
        if not sucesso:
            break
        
        contador_frames_total += 1
        sucesso_rastreio, bbox_atual = tracker.update(frame_atual)

        if sucesso_rastreio:
            centro_atual = obter_centro(bbox_atual)
            pos_x_rel = centro_atual[0] - centro_origem[0]
            pos_y_rel = -(centro_atual[1] - centro_origem[1])
            dados_trajetoria.append([contador_frames_total, pos_x_rel, pos_y_rel, centro_atual[0], centro_atual[1]])

            distancia_percorrida = np.sqrt((centro_atual[0] - posicao_ultimo_carimbo[0])**2 + (centro_atual[1] - posicao_ultimo_carimbo[1])**2)
            if distancia_percorrida >= distancia_minima_para_carimbar:
                (x, y, w, h) = [int(v) for v in bbox_atual]
                regiao_objeto = frame_atual[y:y+h, x:x+w]
                if regiao_objeto.size > 0:
                    imagem_estroboscopica[y:y+h, x:x+w] = regiao_objeto
                posicao_ultimo_carimbo = centro_atual

    captura.release()
    os.remove(video_path) # Limpa o arquivo temporário
    
    # Converte os dados para um DataFrame do Pandas e depois para CSV
    df = pd.DataFrame(dados_trajetoria, columns=['frame', 'pos_x_relativa', 'pos_y_relativa', 'pos_x_absoluta', 'pos_y_absoluta'])
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    # Converte a imagem final para um formato que o Streamlit pode exibir
    _, img_encoded = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = img_encoded.tobytes()

    return img_bytes, csv_bytes

# --- Interface da Aplicação com Streamlit ---

st.set_page_config(layout="wide", page_title="Gerador de Imagem Estroboscópica")

st.title("Gerador de Imagem Estroboscópica e Dados de Trajetória")
st.write("Faça o upload de um vídeo com câmera estática para criar uma imagem que mostra o movimento de um objeto e extrair seus dados de trajetória.")

# Área de upload
video_file = st.file_uploader("1. Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if video_file:
    st.video(video_file)

    # Mostra o primeiro frame para ajudar o usuário a selecionar a BBox
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    success, first_frame = vf.read()
    vf.release()
    os.remove(tfile.name)

    if success:
        st.subheader("2. Defina a área de interesse (Bounding Box)")
        st.write("Use um editor de imagem (como o Paint) para descobrir as coordenadas aproximadas do objeto no primeiro frame, mostrado abaixo:")
        
        # Converte a cor de BGR (OpenCV) para RGB (Streamlit)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        st.image(first_frame_rgb, caption='Primeiro frame do vídeo. A origem (0,0) é o canto superior esquerdo.')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.number_input("X (distância da esquerda)", min_value=0, value=100)
        with col2:
            y = st.number_input("Y (distância do topo)", min_value=0, value=100)
        with col3:
            w = st.number_input("Largura (Width)", min_value=10, value=50)
        with col4:
            h = st.number_input("Altura (Height)", min_value=10, value=50)

        bbox = (x, y, w, h)
        
        # Botão para iniciar o processamento
        if st.button("Gerar Imagem e Dados", type="primary"):
            with st.spinner('Processando o vídeo... Isso pode levar alguns minutos.'):
                # Reseta o ponteiro do arquivo de vídeo para o início
                video_file.seek(0)
                video_bytes = video_file.read()
                
                # Chama a função de processamento
                resultado_img, resultado_csv = processar_video(video_bytes, bbox)

                if resultado_img and resultado_csv:
                    st.subheader("Resultados")
                    st.image(resultado_img, caption='Imagem Estroboscópica Gerada')
                    st.download_button(
                        label="Baixar Imagem (.png)",
                        data=resultado_img,
                        file_name="imagem_estroboscopica.png",
                        mime="image/png"
                    )

                    st.dataframe(pd.read_csv(pd.io.common.BytesIO(resultado_csv)))
                    st.download_button(
                        label="Baixar Dados (.csv)",
                        data=resultado_csv,
                        file_name="dados_trajetoria.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("O processamento falhou. Verifique os parâmetros e o vídeo.")
