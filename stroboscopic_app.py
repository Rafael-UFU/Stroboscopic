import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from io import BytesIO

# --- NOVAS FUNÇÕES AUXILIARES ---

def desenhar_grade_e_eixos(frame, intervalo=100):
    """Desenha uma grade e os valores dos eixos X e Y sobre um frame."""
    frame_com_grade = frame.copy()
    altura, largura, _ = frame_com_grade.shape
    cor_linha = (0, 255, 0)  # Verde
    espessura_linha = 1
    cor_texto = (0, 255, 0)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = 0.5

    # Desenha linhas verticais e rótulos do eixo X
    for x in range(intervalo, largura, intervalo):
        cv2.line(frame_com_grade, (x, 0), (x, altura), cor_linha, espessura_linha)
        cv2.putText(frame_com_grade, str(x), (x + 5, 15), fonte, escala_fonte, cor_texto, 1)

    # Desenha linhas horizontais e rótulos do eixo Y
    for y in range(intervalo, altura, intervalo):
        cv2.line(frame_com_grade, (0, y), (largura, y), cor_linha, espessura_linha)
        cv2.putText(frame_com_grade, str(y), (5, y - 5), fonte, escala_fonte, cor_texto, 1)
        
    return frame_com_grade

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (ATUALIZADA) ---

def processar_video(video_bytes, bbox_coords, fator_distancia, status_text_element):
    """Processa o vídeo, atualizando o status em tempo real."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    captura = cv2.VideoCapture(video_path)
    if not captura.isOpened(): return None, None

    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    sucesso, frame_inicial = captura.read()
    if not sucesso: return None, None

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_inicial, bbox_coords)

    imagem_estroboscopica = frame_inicial.copy()
    altura_frame, largura_frame, _ = frame_inicial.shape
    
    dados_trajetoria = []
    centro_origem = (bbox_coords[0] + bbox_coords[2] / 2, bbox_coords[1] + bbox_coords[3] / 2)
    dados_trajetoria.append([0, 0.0, 0.0, centro_origem[0], centro_origem[1]])
    
    largura_objeto = bbox_coords[2]
    distancia_minima_para_carimbar = largura_objeto * fator_distancia
    posicao_ultimo_carimbo = centro_origem

    contador_frames_total = 0
    carimbos_realizados = 0
    rastreios_sucesso = 0

    while True:
        sucesso, frame_atual = captura.read()
        if not sucesso: break
        
        contador_frames_total += 1
        status_text_element.text(f"Processando frame {contador_frames_total}/{total_frames}...")

        sucesso_rastreio, bbox_atual = tracker.update(frame_atual)

        if sucesso_rastreio:
            rastreios_sucesso += 1
            centro_atual = (bbox_atual[0] + bbox_atual[2] / 2, bbox_atual[1] + bbox_atual[3] / 2)
            
            pos_x_rel = centro_atual[0] - centro_origem[0]
            pos_y_rel = -(centro_atual[1] - centro_origem[1])
            dados_trajetoria.append([contador_frames_total, pos_x_rel, pos_y_rel, centro_atual[0], centro_atual[1]])

            distancia_percorrida = np.sqrt((centro_atual[0] - posicao_ultimo_carimbo[0])**2 + (centro_atual[1] - posicao_ultimo_carimbo[1])**2)
            if distancia_percorrida >= distancia_minima_para_carimbar:
                carimbos_realizados += 1
                (x, y, w, h) = [int(v) for v in bbox_atual]

                x_start, y_start = max(x, 0), max(y, 0)
                x_end, y_end = min(x + w, largura_frame), min(y + h, altura_frame)
                
                regiao_objeto = frame_atual[y_start:y_end, x_start:x_end]
                
                if regiao_objeto.shape[0] > 0 and regiao_objeto.shape[1] > 0:
                    imagem_estroboscopica[y_start:y_end, x_start:x_end] = regiao_objeto
                
                posicao_ultimo_carimbo = centro_atual

    captura.release()
    os.remove(video_path)

    # Fornece um resumo do processamento
    status_text_element.text(f"Processamento concluído! Objeto rastreado em {rastreios_sucesso}/{total_frames} frames. Imagem gerada com {carimbos_realizados} 'carimbos'.")
    
    if carimbos_realizados == 0:
        return None, None # Indica que o resultado não é uma imagem estroboscópica

    df = pd.DataFrame(dados_trajetoria, columns=['frame', 'pos_x_relativa', 'pos_y_relativa', 'pos_x_absoluta', 'pos_y_absoluta'])
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    _, buffer = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = BytesIO(buffer).getvalue()

    return img_bytes, csv_bytes

# --- INTERFACE DA APLICAÇÃO COM STREAMLIT ---

st.set_page_config(layout="wide", page_title="Gerador de Imagem Estroboscópica")
st.title("🔬 Gerador de Imagem Estroboscópica e Dados de Trajetória")
st.write("Faça o upload de um vídeo com câmera estática, defina a área do objeto usando a grade de referência e gere a imagem e os dados de sua trajetória.")

video_file = st.file_uploader("1. Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if video_file:
    # Extrai o primeiro frame para a grade
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    success, first_frame = vf.read()
    vf.release()
    os.remove(tfile.name)

    if success:
        st.subheader("2. Defina a área de interesse (Bounding Box)")
        st.write("Use a grade abaixo para estimar as coordenadas em pixels do seu objeto. **A precisão aqui é crucial para o rastreamento funcionar!**")
        
        # Desenha a grade e exibe a imagem de referência
        frame_com_grade = desenhar_grade_e_eixos(first_frame, intervalo=100)
        frame_com_grade_rgb = cv2.cvtColor(frame_com_grade, cv2.COLOR_BGR2RGB)
        st.image(frame_com_grade_rgb, caption='Frame inicial com grade de referência. Use-o para preencher os campos abaixo.')

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x = st.number_input("Coordenada X (início)", min_value=0, step=10)
        with col2:
            y = st.number_input("Coordenada Y (início)", min_value=0, step=10)
        with col3:
            w = st.number_input("Largura (Width)", min_value=10, step=10)
        with col4:
            h = st.number_input("Altura (Height)", min_value=10, step=10)

        fator_dist = st.slider("Espaçamento entre as imagens", 0.1, 3.0, 0.8, 0.1, help="Valores menores = mais sobreposição. Valores maiores = mais espaço.")

        if st.button("🚀 Gerar Imagem e Dados", type="primary"):
            # Cria um elemento vazio para exibir o status
            status_text = st.empty()
            
            video_file.seek(0)
            video_bytes = video_file.read()
            
            resultado_img, resultado_csv = processar_video(video_bytes, (x, y, w, h), fator_dist, status_text)

            if resultado_img and resultado_csv:
                st.subheader("✅ Resultados")
                st.image(resultado_img, caption='Imagem Estroboscópica Gerada')
                st.download_button("💾 Baixar Imagem (.png)", resultado_img, "imagem_estroboscopica.png", "image/png")
                
                df_resultado = pd.read_csv(BytesIO(resultado_csv))
                st.dataframe(df_resultado)
                st.download_button("💾 Baixar Dados (.csv)", resultado_csv, "dados_trajetoria.csv", "text/csv")
            else:
                st.error("Falha ao gerar a imagem estroboscópica. O rastreador pode ter perdido o objeto. Tente ajustar as coordenadas da área de interesse para que a caixa envolva o objeto de forma mais precisa e tente novamente.")
