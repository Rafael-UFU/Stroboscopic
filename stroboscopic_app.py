import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from io import BytesIO
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# --- FUNÇÕES DE GRÁFICOS ---

def plotar_graficos(df):
    """Cria e exibe os três gráficos de análise de movimento."""
    
    # Configuração geral dos gráficos
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout(pad=5.0)

    # --- Gráfico 1: Trajetória (Y vs X) com Spline ---
    x_coords = df['pos_x_absoluta'].to_numpy()
    y_coords = df['pos_y_absoluta'].to_numpy()
    
    # Inverte o eixo Y para a visualização do gráfico ser mais intuitiva
    ax1.invert_yaxis()
    ax1.scatter(x_coords, y_coords, label='Pontos Observados', color='blue', alpha=0.6, s=10)

    # Cria a curva spline suave
    if len(x_coords) > 3:
        try:
            # Ordena os pontos por X para o spline funcionar corretamente
            sorted_indices = np.argsort(x_coords)
            x_sorted = x_coords[sorted_indices]
            y_sorted = y_coords[sorted_indices]
            
            X_Y_Spline = make_interp_spline(x_sorted, y_sorted)
            X_ = np.linspace(x_sorted.min(), x_sorted.max(), 500)
            Y_ = X_Y_Spline(X_)
            ax1.plot(X_, Y_, label='Curva de Trajetória (Spline)', color='red', linewidth=2)
        except Exception as e:
            # Se o spline falhar (ex: movimento vertical), plota uma linha simples
            ax1.plot(x_coords, y_coords, label='Linha de Trajetória', color='red', linewidth=2, alpha=0.8)

    ax1.set_title('Gráfico de Trajetória', fontsize=16)
    ax1.set_xlabel('Posição X (pixels)')
    ax1.set_ylabel('Posição Y (pixels)')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')

    # --- Gráfico 2: Velocidade vs. Tempo ---
    ax2.plot(df['tempo_s'], df['velocidade_px_s'], label='Velocidade', color='green')
    ax2.set_title('Magnitude da Velocidade vs. Tempo', fontsize=16)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Velocidade (pixels/s)')
    ax2.legend()

    # --- Gráfico 3: Aceleração vs. Tempo ---
    ax3.plot(df['tempo_s'], df['aceleracao_px_s2'], label='Aceleração', color='purple')
    ax3.set_title('Magnitude da Aceleração vs. Tempo', fontsize=16)
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Aceleração (pixels/s²)')
    ax3.legend()

    return fig

# --- FUNÇÕES AUXILIARES ---

def desenhar_grade_cartesiana(frame, intervalo=100):
    """Desenha uma grade com a origem (0,0) no canto INFERIOR ESQUERDO."""
    frame_com_grade = frame.copy()
    altura, largura, _ = frame_com_grade.shape
    cor_linha, cor_texto = (0, 255, 0, 200), (0, 255, 0)
    fonte, escala_fonte = cv2.FONT_HERSHEY_SIMPLEX, 0.5

    for x in range(intervalo, largura, intervalo):
        cv2.line(frame_com_grade, (x, 0), (x, altura), cor_linha, 1)
        cv2.putText(frame_com_grade, str(x), (x - 10, altura - 10), fonte, escala_fonte, cor_texto, 1)
    for y in range(intervalo, altura, intervalo):
        pos_y_imagem = altura - y
        cv2.line(frame_com_grade, (0, pos_y_imagem), (largura, pos_y_imagem), cor_linha, 1)
        cv2.putText(frame_com_grade, str(y), (10, pos_y_imagem + 5), fonte, escala_fonte, cor_texto, 1)
    return frame_com_grade

# --- FUNÇÃO DE PROCESSAMENTO PRINCIPAL (ATUALIZADA) ---

def processar_video(video_bytes, bbox_coords_opencv, fator_distancia, status_text_element):
    """Processa o vídeo, calcula a cinemática e retorna a imagem, dados e gráficos."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    captura = cv2.VideoCapture(video_path)
    if not captura.isOpened(): return None, None, None

    fps = captura.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        st.error("Não foi possível determinar o FPS do vídeo. Usando 30 FPS como padrão.")
        fps = 30
    
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    sucesso, frame_inicial = captura.read()
    if not sucesso: return None, None, None

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_inicial, bbox_coords_opencv)

    imagem_estroboscopica = frame_inicial.copy()
    altura_frame, largura_frame, _ = frame_inicial.shape
    
    raw_data = [] # Lista para armazenar dados brutos (frame, x, y)
    
    largura_objeto = bbox_coords_opencv[2]
    distancia_minima_para_carimbar = largura_objeto * fator_distancia
    centro_origem = (bbox_coords_opencv[0] + bbox_coords_opencv[2] / 2, bbox_coords_opencv[1] + bbox_coords_opencv[3] / 2)
    posicao_ultimo_carimbo = centro_origem

    contador_frames_total, carimbos_realizados = 0, 0
    while True:
        sucesso, frame_atual = captura.read()
        if not sucesso: break
        
        contador_frames_total += 1
        status_text_element.text(f"Processando frame {contador_frames_total}/{total_frames}...")

        sucesso_rastreio, bbox_atual = tracker.update(frame_atual)

        if sucesso_rastreio:
            centro_atual = (bbox_atual[0] + bbox_atual[2] / 2, bbox_atual[1] + bbox_atual[3] / 2)
            raw_data.append([contador_frames_total, centro_atual[0], centro_atual[1]])
            
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

    if not raw_data:
        status_text_element.error("O rastreador não conseguiu seguir o objeto em nenhum frame.")
        return None, None, None
    
    # --- Pós-processamento dos dados para calcular cinemática ---
    df = pd.DataFrame(raw_data, columns=['frame', 'pos_x_absoluta', 'pos_y_absoluta'])
    df['tempo_s'] = df['frame'] / fps
    
    # Posições relativas
    df['pos_x_relativa'] = df['pos_x_absoluta'] - centro_origem[0]
    df['pos_y_relativa'] = -(df['pos_y_absoluta'] - centro_origem[1])
    
    # Cálculo da velocidade
    delta_x = df['pos_x_absoluta'].diff()
    delta_y = df['pos_y_absoluta'].diff()
    delta_t = df['tempo_s'].diff()
    df['velocidade_px_s'] = np.sqrt(delta_x**2 + delta_y**2) / delta_t
    
    # Suavização da velocidade para um cálculo de aceleração mais estável
    # Usa um filtro Savitzky-Golay. window_length deve ser ímpar e menor que os dados.
    window_len = min(51, len(df) - 2 if len(df) % 2 == 0 else len(df) - 1)
    if window_len > 3: # O filtro precisa de um número mínimo de pontos
        df['velocidade_suavizada'] = savgol_filter(df['velocidade_px_s'].fillna(0), window_len, 3)
    else:
        df['velocidade_suavizada'] = df['velocidade_px_s']

    # Cálculo da aceleração a partir da velocidade suavizada
    delta_v = df['velocidade_suavizada'].diff()
    df['aceleracao_px_s2'] = delta_v / delta_t
    
    # Limpa colunas auxiliares e preenche valores nulos (NaN)
    df_final = df[['frame', 'tempo_s', 'pos_x_relativa', 'pos_y_relativa', 'pos_x_absoluta', 'pos_y_absoluta', 'velocidade_px_s', 'aceleracao_px_s2']].copy()
    df_final = df_final.fillna(0)

    status_text_element.success(f"Processamento concluído! Análise realizada em {len(df_final)} frames.")
    
    # Gera imagem e CSV
    csv_bytes = df_final.to_csv(index=False).encode('utf-8')
    _, buffer = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = BytesIO(buffer).getvalue()
    
    # Gera a figura com os gráficos
    figura_graficos = plotar_graficos(df_final)

    return img_bytes, csv_bytes, figura_graficos

# --- INTERFACE DA APLICAÇÃO ---

st.set_page_config(layout="wide", page_title="Análise de Movimento por Vídeo")
st.markdown("# 🔬 Análise de Movimento por Vídeo")
st.markdown("### Crie imagens estroboscópicas, extraia dados de trajetória e visualize gráficos de cinemática.")

video_file = st.file_uploader("1. Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if video_file:
    # O resto da interface continua muito similar...
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    success, first_frame = vf.read()
    vf.release()
    os.remove(tfile.name)

    if success:
        st.markdown("## 2. Defina a Área de Interesse (Bounding Box)")
        st.info("Utilize a grade de referência para preencher os campos. O retângulo azul será atualizado em tempo real.")
        
        altura_total, _, _ = first_frame.shape

        col_config, col_preview = st.columns([1, 2])

        with col_config:
            st.markdown("#### Parâmetros de Seleção")
            x = st.number_input("Coordenada X (do canto esquerdo)", min_value=0, step=10)
            y_usuario = st.number_input("Coordenada Y (do canto inferior)", min_value=0, step=10)
            w = st.number_input("Largura (Width)", min_value=10, value=50, step=10)
            h = st.number_input("Altura (Height)", min_value=10, value=50, step=10)
            
            st.markdown("#### Parâmetros de Geração")
            fator_dist = st.slider("Espaçamento na Imagem Estroboscópica", 0.1, 3.0, 0.8, 0.1, help="Define o quão longe o objeto precisa se mover para ser 'carimbado' na imagem final.")

            y_opencv = altura_total - y_usuario - h
            bbox_opencv = (x, y_opencv, w, h)
            
            if st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):
                status_text = st.empty()
                video_file.seek(0)
                video_bytes = video_file.read()
                
                resultado_img, resultado_csv, figura_graficos = processar_video(video_bytes, bbox_opencv, fator_dist, status_text)

                if resultado_img and resultado_csv and figura_graficos:
                    st.markdown("## ✅ Resultados da Análise")
                    
                    st.markdown("### Imagem Estroboscópica")
                    st.image(resultado_img, caption='Trajetória do objeto visualizada na imagem.')
                    st.download_button("💾 Baixar Imagem (.png)", resultado_img, "imagem_estroboscopica.png", "image/png", use_container_width=True)
                    
                    st.markdown("### Gráficos de Cinemática")
                    st.pyplot(figura_graficos)
                    
                    st.markdown("### Tabela de Dados Completa")
                    df_resultado = pd.read_csv(BytesIO(resultado_csv))
                    st.dataframe(df_resultado)
                    st.download_button("💾 Baixar Dados (.csv)", resultado_csv, "dados_trajetoria.csv", "text/csv", use_container_width=True)
                else:
                    st.error("Falha na análise. O rastreador pode ter perdido o objeto. Tente ajustar as coordenadas da área de interesse para que o retângulo azul envolva o objeto de forma mais precisa e tente novamente.")
        
        with col_preview:
            frame_com_grade = desenhar_grade_cartesiana(first_frame, intervalo=100)
            if w > 0 and h > 0:
                cv2.rectangle(frame_com_grade, (x, y_opencv), (x + w, y_opencv + h), (255, 0, 0), 2)
            frame_com_grade_rgb = cv2.cvtColor(frame_com_grade, cv2.COLOR_BGR2RGB)
            st.image(frame_com_grade_rgb, caption='Use esta referência para definir a área do objeto. A origem (0,0) está no canto inferior esquerdo.', use_column_width=True)
