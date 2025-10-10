import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from io import BytesIO

# --- Funções de Lógica (Revisadas e Corrigidas) ---

def obter_centro(bbox):
    """Calcula o ponto central de uma bounding box (x, y, w, h)."""
    x, y, w, h = [int(v) for v in bbox]
    return (x + w / 2, y + h / 2)

def processar_video(video_bytes, bbox_coords, fator_distancia, progresso_bar):
    """
    Função principal que processa o vídeo e retorna a imagem e os dados.
    """
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    captura = cv2.VideoCapture(video_path)
    if not captura.isOpened():
        st.error("Erro ao abrir o arquivo de vídeo.")
        return None, None

    # Obter total de frames para a barra de progresso
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))

    sucesso, frame_inicial = captura.read()
    if not sucesso:
        st.error("Não foi possível ler o primeiro frame do vídeo.")
        return None, None
    
    # Inicializa o tracker com as coordenadas recebidas da interface
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_inicial, bbox_coords)

    # A imagem estroboscópica começa como o primeiro frame
    imagem_estroboscopica = frame_inicial.copy()
    altura_frame, largura_frame, _ = frame_inicial.shape
    
    # Lógica de dados CSV
    dados_trajetoria = []
    centro_origem = obter_centro(bbox_coords)
    dados_trajetoria.append([0, 0.0, 0.0, centro_origem[0], centro_origem[1]])
    
    # Lógica de "carimbo" adaptativo
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
            
            # Gravação dos dados CSV
            pos_x_rel = centro_atual[0] - centro_origem[0]
            pos_y_rel = -(centro_atual[1] - centro_origem[1])
            dados_trajetoria.append([contador_frames_total, pos_x_rel, pos_y_rel, centro_atual[0], centro_atual[1]])

            # Lógica da Imagem Estroboscópica
            distancia_percorrida = np.sqrt((centro_atual[0] - posicao_ultimo_carimbo[0])**2 + (centro_atual[1] - posicao_ultimo_carimbo[1])**2)
            if distancia_percorrida >= distancia_minima_para_carimbar:
                (x, y, w, h) = [int(v) for v in bbox_atual]

                # --- CORREÇÃO CRÍTICA AQUI ---
                # Garante que as coordenadas não saiam dos limites do frame
                x_start, y_start = max(x, 0), max(y, 0)
                x_end, y_end = min(x + w, largura_frame), min(y + h, altura_frame)
                
                # Pega a região do objeto do frame ATUAL
                regiao_objeto = frame_atual[y_start:y_end, x_start:x_end]
                
                # Cola a região na imagem estroboscópica, garantindo que as dimensões batam
                if regiao_objeto.shape[0] > 0 and regiao_objeto.shape[1] > 0:
                    imagem_estroboscopica[y_start:y_end, x_start:x_end] = regiao_objeto
                
                posicao_ultimo_carimbo = centro_atual
        
        # Atualiza a barra de progresso
        progresso_bar.progress(contador_frames_total / total_frames)


    captura.release()
    os.remove(video_path)
    
    # Finaliza a barra de progresso
    progresso_bar.progress(1.0)

    # Converte os dados para um DataFrame do Pandas e depois para CSV
    df = pd.DataFrame(dados_trajetoria, columns=['frame', 'pos_x_relativa', 'pos_y_relativa', 'pos_x_absoluta', 'pos_y_absoluta'])
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    # Converte a imagem final para um formato de bytes PNG
    _, buffer = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = BytesIO(buffer).getvalue()

    return img_bytes, csv_bytes

# --- Interface da Aplicação com Streamlit ---

st.set_page_config(layout="wide", page_title="Gerador de Imagem Estroboscópica")

st.title("🔬 Gerador de Imagem Estroboscópica e Dados de Trajetória")
st.write("Faça o upload de um vídeo com câmera estática para criar uma imagem que mostra o movimento de um objeto e extrair seus dados de trajetória.")

# Área de upload
video_file = st.file_uploader("1. Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if video_file:
    # Mostra o primeiro frame para ajudar o usuário a selecionar a BBox
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    success, first_frame = vf.read()
    vf.release()
    os.remove(tfile.name)

    if success:
        st.subheader("2. Defina a área de interesse (Bounding Box)")
        st.write("Use um editor de imagem (como o Paint ou a ferramenta de Captura de Tela do seu SO) para descobrir as coordenadas aproximadas do objeto no primeiro frame. A origem (0,0) é o canto superior esquerdo.")
        
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        st.image(first_frame_rgb, caption='Primeiro frame do vídeo para referência.')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.number_input("X (distância da esquerda)", min_value=0, value=100, step=1)
        with col2:
            y = st.number_input("Y (distância do topo)", min_value=0, value=100, step=1)
        with col3:
            w = st.number_input("Largura (Width)", min_value=10, value=50, step=1)
        with col4:
            h = st.number_input("Altura (Height)", min_value=10, value=50, step=1)

        fator_dist = st.slider("Fator de Distância (espaçamento)", min_value=0.1, max_value=2.0, value=0.8, step=0.1,
                               help="Valores menores = mais 'cópias' do objeto. Valores maiores = mais espaçamento.")

        bbox = (x, y, w, h)
        
        if st.button("🚀 Gerar Imagem e Dados", type="primary"):
            barra_de_progresso = st.progress(0, text="Iniciando processamento...")
            
            video_file.seek(0)
            video_bytes = video_file.read()
            
            resultado_img, resultado_csv = processar_video(video_bytes, bbox, fator_dist, barra_de_progresso)

            if resultado_img and resultado_csv:
                st.subheader("✅ Resultados")
                st.image(resultado_img, caption='Imagem Estroboscópica Gerada')
                st.download_button(
                    label="💾 Baixar Imagem (.png)",
                    data=resultado_img,
                    file_name="imagem_estroboscopica.png",
                    mime="image/png"
                )
                
                df_resultado = pd.read_csv(BytesIO(resultado_csv))
                st.dataframe(df_resultado)
                st.download_button(
                    label="💾 Baixar Dados (.csv)",
                    data=resultado_csv,
                    file_name="dados_trajetoria.csv",
                    mime="text/csv"
                )
            else:
                st.error("O processamento falhou. Verifique os parâmetros e o vídeo.")
