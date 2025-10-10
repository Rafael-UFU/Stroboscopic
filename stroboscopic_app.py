import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from io import BytesIO

# --- FUNÇÕES AUXILIARES (ATUALIZADAS) ---

def desenhar_grade_cartesiana(frame, intervalo=100):
    """
    Desenha uma grade com a origem (0,0) no canto INFERIOR ESQUERDO.
    """
    frame_com_grade = frame.copy()
    altura, largura, _ = frame_com_grade.shape
    cor_linha = (0, 255, 0, 200)  # Verde com alguma transparência
    cor_texto = (0, 255, 0)
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_fonte = 0.5

    # Desenha linhas verticais (Eixo X) - Origem na esquerda
    for x in range(intervalo, largura, intervalo):
        cv2.line(frame_com_grade, (x, 0), (x, altura), cor_linha, 1)
        cv2.putText(frame_com_grade, str(x), (x - 10, altura - 10), fonte, escala_fonte, cor_texto, 1)

    # Desenha linhas horizontais (Eixo Y) - Origem embaixo
    for y in range(intervalo, altura, intervalo):
        # A linha é desenhada na coordenada de imagem (top-left)
        pos_y_imagem = altura - y
        cv2.line(frame_com_grade, (0, pos_y_imagem), (largura, pos_y_imagem), cor_linha, 1)
        cv2.putText(frame_com_grade, str(y), (10, pos_y_imagem + 5), fonte, escala_fonte, cor_texto, 1)
        
    return frame_com_grade

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (SEM ALTERAÇÕES NA LÓGICA) ---

def processar_video(video_bytes, bbox_coords_opencv, fator_distancia, status_text_element):
    """Processa o vídeo. Recebe as coordenadas já convertidas para o sistema do OpenCV."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    captura = cv2.VideoCapture(video_path)
    if not captura.isOpened(): return None, None

    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    sucesso, frame_inicial = captura.read()
    if not sucesso: return None, None

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_inicial, bbox_coords_opencv)

    imagem_estroboscopica = frame_inicial.copy()
    altura_frame, largura_frame, _ = frame_inicial.shape
    
    dados_trajetoria = []
    centro_origem = (bbox_coords_opencv[0] + bbox_coords_opencv[2] / 2, bbox_coords_opencv[1] + bbox_coords_opencv[3] / 2)
    dados_trajetoria.append([0, 0.0, 0.0, centro_origem[0], centro_origem[1]])
    
    largura_objeto = bbox_coords_opencv[2]
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
            
            # A conversão para o CSV já estava correta para um eixo Y cartesiano
            pos_x_rel = centro_atual[0] - centro_origem[0]
            pos_y_rel = -(centro_atual[1] - centro_origem[1])
            dados_trajetoria.append([contador_frames_total, pos_x_rel, pos_y_rel, centro_atual[0], centro_atual[1]])

            distancia_percorrida = np.sqrt((centro_atual[0] - posicao_ultimo_carimbo[0])**2 + (centro_atual[1] - posicao_ultimo_carimbo[1])**2)
            if distancia_percorrida >= distancia_minima_para_carimbar:
                carimbos_realizados += 1
                (x, y, w, h) = [int(v) for v in bbox_atual]
                x_start, y_start, x_end, y_end = max(x, 0), max(y, 0), min(x + w, largura_frame), min(y + h, altura_frame)
                regiao_objeto = frame_atual[y_start:y_end, x_start:x_end]
                
                if regiao_objeto.shape[0] > 0 and regiao_objeto.shape[1] > 0:
                    imagem_estroboscopica[y_start:y_end, x_start:x_end] = regiao_objeto
                posicao_ultimo_carimbo = centro_atual

    captura.release()
    os.remove(video_path)

    status_text_element.text(f"Processamento concluído! Objeto rastreado em {rastreios_sucesso}/{total_frames} frames. Imagem gerada com {carimbos_realizados} 'carimbos'.")
    
    if carimbos_realizados == 0:
        return None, None

    df = pd.DataFrame(dados_trajetoria, columns=['frame', 'pos_x_relativa', 'pos_y_relativa', 'pos_x_absoluta', 'pos_y_absoluta'])
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    
    _, buffer = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = BytesIO(buffer).getvalue()

    return img_bytes, csv_bytes

# --- INTERFACE DA APLICAÇÃO COM STREAMLIT (ATUALIZADA) ---

st.set_page_config(layout="wide", page_title="Gerador de Imagem Estroboscópica")
st.markdown("# 🔬 Gerador de Imagem Estroboscópica")
st.markdown("### Crie imagens de trajetória e extraia dados de movimento a partir de vídeos.")

video_file = st.file_uploader("1. Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    success, first_frame = vf.read()
    vf.release()
    os.remove(tfile.name)

    if success:
        st.markdown("## 2. Defina a Área de Interesse/Alvo (Bounding Box)")
        st.info("Utilize a grade de referência abaixo para preencher os campos. **O retângulo azul será atualizado em tempo real.**")
        
        altura_total, _, _ = first_frame.shape

        col_config, col_preview = st.columns([1, 2])

        with col_config:
            st.markdown("#### Parâmetros de Seleção")
            x = st.number_input("Coordenada X (do canto esquerdo)", min_value=0, step=10)
            y_usuario = st.number_input("Coordenada Y (do canto inferior)", min_value=0, step=10)
            w = st.number_input("Largura (Width)", min_value=10, value=50, step=10)
            h = st.number_input("Altura (Height)", min_value=10, value=50, step=10)
            
            st.markdown("#### Parâmetros de Geração")
            fator_dist = st.slider("Espaçamento entre as imagens", 0.1, 3.0, 0.8, 0.1, help="Valores menores = mais sobreposição. Valores maiores = mais espaço.")

            # --- Conversão de Coordenadas ---
            # O usuário insere Y a partir de baixo, mas o OpenCV precisa de Y a partir de cima.
            y_opencv = altura_total - y_usuario - h
            bbox_opencv = (x, y_opencv, w, h)
            
            if st.button("🚀 Gerar Imagem e Dados", type="primary", use_container_width=True):
                status_text = st.empty()
                video_file.seek(0)
                video_bytes = video_file.read()
                
                resultado_img, resultado_csv = processar_video(video_bytes, bbox_opencv, fator_dist, status_text)

                if resultado_img and resultado_csv:
                    st.markdown("## ✅ Resultados")
                    st.image(resultado_img, caption='Imagem Estroboscópica Gerada')
                    st.download_button("💾 Baixar Imagem (.png)", resultado_img, "imagem_estroboscopica.png", "image/png", use_container_width=True)
                    
                    df_resultado = pd.read_csv(BytesIO(resultado_csv))
                    st.dataframe(df_resultado)
                    st.download_button("💾 Baixar Dados (.csv)", resultado_csv, "dados_trajetoria.csv", "text/csv", use_container_width=True)
                else:
                    st.error("Falha ao gerar a imagem. O rastreador pode ter perdido o objeto. Tente ajustar as coordenadas da área de interesse para que o retângulo azul envolva o objeto de forma mais precisa.")
        
        with col_preview:
            # Desenha a grade com o novo sistema de coordenadas
            frame_com_grade = desenhar_grade_cartesiana(first_frame, intervalo=100)
            
            # Desenha o retângulo de pré-visualização
            if w > 0 and h > 0:
                cv2.rectangle(frame_com_grade, (x, y_opencv), (x + w, y_opencv + h), (255, 0, 0), 2) # Azul em BGR

            frame_com_grade_rgb = cv2.cvtColor(frame_com_grade, cv2.COLOR_BGR2RGB)
            st.image(frame_com_grade_rgb, caption='Use esta referência para definir a área do objeto. A origem (0,0) está no canto inferior esquerdo.', use_column_width=True)
