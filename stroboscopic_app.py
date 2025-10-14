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
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- CSS PARA SOBREPOR O CANVAS ---
st.markdown("""
<style>
/* Estilo para o container da imagem e canvas */
.image-and-canvas-container {
    position: relative;
    display: inline-block; /* Para que o container se ajuste à imagem */
}

/* Estilo para a imagem dentro do container */
.image-and-canvas-container img {
    display: block; /* Remove espaços extras abaixo da imagem */
}

/* Estilo para o canvas dentro do container, sobreposto à imagem */
.image-and-canvas-container canvas {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 1; /* Garante que o canvas fique por cima da imagem */
}
</style>
""", unsafe_allow_html=True)


# --- FUNÇÕES DE LÓGICA E PROCESSAMENTO ---
# (As funções de plotar gráficos e processar vídeo foram movidas para o final para melhor organização)

# --- INTERFACE DA APLICAÇÃO ---

st.set_page_config(layout="wide", page_title="Análise de Movimento por Vídeo")
st.markdown("# 🔬 Análise de Movimento por Vídeo")
st.markdown("### Uma ferramenta para extrair dados cinemáticos de vídeos com câmera estática.")

# Inicializa o estado da sessão para controlar o fluxo
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'initial_frame' not in st.session_state:
    st.session_state.initial_frame = None
if 'video_bytes' not in st.session_state:
    st.session_state.video_bytes = None
if 'scale_factor' not in st.session_state:
    st.session_state.scale_factor = None
if 'origin_coords' not in st.session_state:
    st.session_state.origin_coords = None

# --- PASSO 0: UPLOAD ---
if st.session_state.step == "upload":
    st.markdown("## Passo 1: Upload do Vídeo")
    video_file = st.file_uploader("Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    
    if video_file:
        st.session_state.video_bytes = video_file.getvalue()
        st.session_state.step = "frame_selection"
        st.rerun()

# --- PASSO 1: SELEÇÃO DO FRAME INICIAL ---
if st.session_state.step == "frame_selection":
    st.markdown("## Passo 2: Seleção do Frame Inicial")
    st.info("Navegue pelos frames para escolher o momento exato em que a análise deve começar.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(st.session_state.video_bytes)
    video_path = tfile.name
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if 'current_frame_idx' not in st.session_state:
        st.session_state.current_frame_idx = 0

    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        if st.button("<< Anterior"):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
    with col3:
        if st.button("Próximo >>"):
            st.session_state.current_frame_idx = min(total_frames - 1, st.session_state.current_frame_idx + 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
    success, frame = cap.read()
    
    with col2:
        if success:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Exibindo Frame: {st.session_state.current_frame_idx} / {total_frames-1}")
        
    if st.button("Confirmar Frame e Iniciar Calibração", type="primary"):
        st.session_state.initial_frame = frame
        st.session_state.step = "calibration"
        st.rerun()
    
    cap.release()
    os.remove(video_path)

# --- PASSO 2: CALIBRAÇÃO DA ESCALA ---
if st.session_state.step == "calibration":
    st.markdown("## Passo 3: Calibração da Escala")
    st.info("Desenhe uma linha sobre um objeto de comprimento conhecido na cena e informe o seu tamanho real.")

    bg_image_calib_np = cv2.cvtColor(st.session_state.initial_frame, cv2.COLOR_BGR2RGB)
    bg_image_calib = Image.fromarray(bg_image_calib_np)
    altura, largura = bg_image_calib.height, bg_image_calib.width

    col_canvas_calib, col_input_calib = st.columns(2)

    with col_canvas_calib:
        st.write("1. Desenhe a linha de referência na imagem:")
        with st.container():
            st.markdown('<div class="image-and-canvas-container">', unsafe_allow_html=True)
            st.image(bg_image_calib) # <-- CORREÇÃO AQUI
            canvas_result_calib = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#FF0000",
                background_color="rgba(0, 0, 0, 0)",
                update_streamlit=True,
                height=altura,
                width=largura,
                drawing_mode="line",
                key="canvas_calib",
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with col_input_calib:
        if canvas_result_calib.json_data is not None and canvas_result_calib.json_data["objects"]:
            st.write("2. Informe o comprimento real da linha:")
            length_real = st.number_input("Comprimento real (em metros)", min_value=0.01, format="%.4f")

            line = canvas_result_calib.json_data["objects"][0]
            x1, y1, x2, y2 = line["left"], line["top"], line["x2"], line["y2"]
            length_pixels = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            st.write(f"Distância em pixels da linha desenhada: **{length_pixels:.2f} px**")

            if st.button("Confirmar Calibração e Definir Origem", type="primary"):
                st.session_state.scale_factor = length_real / length_pixels
                st.success(f"Fator de Escala calculado: {st.session_state.scale_factor:.6f} m/pixel")
                st.session_state.step = "origin_setting"
                st.rerun()

# --- PASSO 3: DEFINIÇÃO DA ORIGEM ---
if st.session_state.step == "origin_setting":
    st.markdown("## Passo 4: Definição da Origem (0, 0)")
    st.info("Clique no ponto da imagem que será a origem do seu sistema de coordenadas.")

    bg_image_origin_np = cv2.cvtColor(st.session_state.initial_frame, cv2.COLOR_BGR2RGB)
    bg_image_origin = Image.fromarray(bg_image_origin_np)
    altura, largura = bg_image_origin.height, bg_image_origin.width

    col_canvas_origin, col_input_origin = st.columns(2)

    with col_canvas_origin:
        st.write("Clique no ponto de origem:")
        with st.container():
            st.markdown('<div class="image-and-canvas-container">', unsafe_allow_html=True)
            st.image(bg_image_origin) # <-- CORREÇÃO AQUI
            canvas_result_origin = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#00FF00",
                background_color="rgba(0, 0, 0, 0)",
                update_streamlit=True,
                height=altura,
                width=largura,
                drawing_mode="point",
                point_display_radius=5,
                key="canvas_origin",
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_input_origin:
        if canvas_result_origin.json_data is not None and canvas_result_origin.json_data["objects"]:
            point = canvas_result_origin.json_data["objects"][0]
            x, y = point["left"], point["top"]
            st.write(f"Origem selecionada (em pixels): **(X: {x}, Y: {y})**")

            if st.button("Confirmar Origem e Selecionar Objeto", type="primary"):
                st.session_state.origin_coords = (x, y)
                st.success("Origem definida com sucesso!")
                st.session_state.step = "roi_selection"
                st.rerun()

# --- PASSO 4: SELEÇÃO DO OBJETO (ROI) ---
if st.session_state.step == "roi_selection":
    st.markdown("## Passo 5: Seleção do Objeto a ser Rastreado")
    st.info("Use a grade de referência e a origem marcada para definir a área inicial do objeto.")

    frame_com_grade = desenhar_grade_cartesiana(st.session_state.initial_frame, intervalo=100)
    # Desenha o marcador da origem
    orig_x, orig_y = int(st.session_state.origin_coords[0]), int(st.session_state.origin_coords[1])
    cv2.circle(frame_com_grade, (orig_x, orig_y), 10, (255, 0, 255), -1)
    cv2.putText(frame_com_grade, "(0,0)", (orig_x + 15, orig_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    altura_total, _, _ = frame_com_grade.shape
    col_config, col_preview = st.columns([1, 2])

    with col_config:
        st.markdown("#### Parâmetros de Seleção")
        x = st.number_input("Coordenada X (do canto esquerdo do objeto)", min_value=0, step=10)
        y_usuario = st.number_input("Coordenada Y (do canto INFERIOR do objeto)", min_value=0, step=10)
        w = st.number_input("Largura (Width)", min_value=10, value=50, step=10)
        h = st.number_input("Altura (Height)", min_value=10, value=50, step=10)
        
        y_opencv = altura_total - y_usuario - h
        bbox_opencv = (x, y_opencv, w, h)
        
        st.markdown("#### Parâmetros de Geração")
        fator_dist = st.slider("Espaçamento na Imagem (metros)", 0.01, 2.0, 0.1, 0.01, help="Distância MÍNIMA (em metros) que o objeto precisa se mover para ser 'carimbado' na imagem final.")

        if st.button("🚀 Iniciar Análise Completa", type="primary", use_container_width=True):
            st.session_state.bbox = bbox_opencv
            st.session_state.fator_dist = fator_dist
            st.session_state.step = "processing"
            st.rerun()
            
    with col_preview:
        frame_para_preview = frame_com_grade.copy()
        if w > 0 and h > 0:
            cv2.rectangle(frame_para_preview, (x, y_opencv), (x + w, y_opencv + h), (255, 0, 0), 2)
        
        st.image(cv2.cvtColor(frame_para_preview, cv2.COLOR_BGR2RGB), caption='Ajuste os valores até o retângulo azul envolver seu objeto.', use_container_width=True)

# --- PASSO 5: PROCESSAMENTO E RESULTADOS ---
if st.session_state.step == "processing":
    st.markdown("## ✅ Resultados da Análise")
    status_text = st.empty()
    
    resultado_img, resultado_csv, figura_graficos = processar_video(
        st.session_state.video_bytes,
        st.session_state.initial_frame,
        st.session_state.current_frame_idx,
        st.session_state.bbox,
        st.session_state.fator_dist,
        st.session_state.scale_factor,
        st.session_state.origin_coords,
        status_text
    )

    if resultado_img and resultado_csv and figura_graficos:
        st.markdown("### Imagem Estroboscópica")
        st.image(resultado_img)
        st.download_button("💾 Baixar Imagem (.png)", resultado_img, "imagem_estroboscopica.png", "image/png", use_container_width=True)
        
        st.markdown("### Gráficos de Cinemática")
        st.pyplot(figura_graficos)
        
        st.markdown("### Tabela de Dados Completa")
        df_resultado = pd.read_csv(BytesIO(resultado_csv))
        st.dataframe(df_resultado)
        st.download_button("💾 Baixar Dados (CSV)", resultado_csv, "dados_trajetoria.csv", "text/csv", use_container_width=True)
        
        if st.button("Analisar outro vídeo"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    else:
        st.error("Falha na análise. O rastreador pode ter perdido o objeto.")

# --- FUNÇÕES DE PLOTAGEM E PROCESSAMENTO ---
def plotar_graficos(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout(pad=5.0)

    x, y = df['pos_x_m'].to_numpy(), df['pos_y_m'].to_numpy()
    ax1.scatter(x, y, label='Pontos Observados', color='blue', alpha=0.6, s=10)
    if len(x) > 3:
        try:
            sorted_indices = np.argsort(x)
            x_s, y_s = x[sorted_indices], y[sorted_indices]
            X_Y_Spline = make_interp_spline(x_s, y_s)
            X_, Y_ = np.linspace(x_s.min(), x_s.max(), 500), X_Y_Spline(np.linspace(x_s.min(), x_s.max(), 500))
            ax1.plot(X_, Y_, label='Curva de Trajetória (Spline)', color='red', linewidth=2)
        except:
            ax1.plot(x, y, label='Linha de Trajetória', color='red', linewidth=2, alpha=0.8)
    ax1.set_title('Gráfico de Trajetória', fontsize=16)
    ax1.set_xlabel('Posição X (m)')
    ax1.set_ylabel('Posição Y (m)')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')

    ax2.plot(df['tempo_s'], df['velocidade_m_s'], label='Velocidade', color='green')
    ax2.set_title('Magnitude da Velocidade vs. Tempo', fontsize=16)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Velocidade (m/s)')
    ax2.legend()

    ax3.plot(df['tempo_s'], df['aceleracao_m_s2'], label='Aceleração', color='purple')
    ax3.set_title('Magnitude da Aceleração vs. Tempo', fontsize=16)
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Aceleração (m/s²)')
    ax3.legend()

    return fig

def desenhar_grade_cartesiana(frame, intervalo=100):
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

def processar_video(video_bytes, initial_frame, start_frame_idx, bbox_coords_opencv, fator_distancia, scale_factor, origin_coords, status_text_element):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    tracker = cv2.TrackerCSRT_create()
    tracker.init(initial_frame, bbox_coords_opencv)

    imagem_estroboscopica = initial_frame.copy()
    altura_frame, largura_frame, _ = initial_frame.shape
    
    raw_data = []
    posicao_ultimo_carimbo = (bbox_coords_opencv[0] + bbox_coords_opencv[2]/2, bbox_coords_opencv[1] + bbox_coords_opencv[3]/2)

    contador_frames_processados = 0
    while True:
        frame_atual_idx = start_frame_idx + contador_frames_processados
        if frame_atual_idx >= total_frames: break

        success, frame_atual = cap.read()
        if not success: break
        
        status_text_element.text(f"Processando frame {frame_atual_idx}/{total_frames-1}...")
        
        success_track, bbox_atual = tracker.update(frame_atual)
        if success_track:
            centro_atual = (bbox_atual[0] + bbox_atual[2]/2, bbox_atual[1] + bbox_atual[3]/2)
            raw_data.append([frame_atual_idx, centro_atual[0], centro_atual[1]])
            
            dist_pixels = np.sqrt((centro_atual[0] - posicao_ultimo_carimbo[0])**2 + (centro_atual[1] - posicao_ultimo_carimbo[1])**2)
            if dist_pixels * scale_factor >= fator_distancia:
                (x, y, w, h) = [int(v) for v in bbox_atual]
                x_s, y_s, x_e, y_e = max(x, 0), max(y, 0), min(x + w, largura_frame), min(y + h, altura_frame)
                regiao = frame_atual[y_s:y_e, x_s:x_e]
                if regiao.size > 0:
                    imagem_estroboscopica[y_s:y_e, x_s:x_e] = regiao
                posicao_ultimo_carimbo = centro_atual
        contador_frames_processados += 1
    
    cap.release()
    os.remove(video_path)
    
    if not raw_data: return None, None, None
    
    df = pd.DataFrame(raw_data, columns=['frame', 'pos_x_px', 'pos_y_px'])
    df['tempo_s'] = (df['frame'] - start_frame_idx) / fps
    
    df['pos_x_m'] = (df['pos_x_px'] - origin_coords[0]) * scale_factor
    df['pos_y_m'] = -(df['pos_y_px'] - origin_coords[1]) * scale_factor
    
    df['velocidade_m_s'] = np.sqrt(df['pos_x_m'].diff()**2 + df['pos_y_m'].diff()**2) / df['tempo_s'].diff()
    window_len = min(51, len(df) - 2 if len(df) % 2 == 0 else len(df) - 1)
    if window_len > 3:
        df['vel_suavizada'] = savgol_filter(df['velocidade_m_s'].fillna(0), window_len, 3)
    else:
        df['vel_suavizada'] = df['velocidade_m_s']
    df['aceleracao_m_s2'] = df['vel_suavizada'].diff() / df['tempo_s'].diff()
    
    df_final = df[['frame', 'tempo_s', 'pos_x_m', 'pos_y_m', 'velocidade_m_s', 'aceleracao_m_s2']].copy().fillna(0)

    status_text_element.success(f"Processamento concluído!")
    
    csv_bytes = df_final.to_csv(index=False).encode('utf-8')
    _, buffer = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = BytesIO(buffer).getvalue()
    figura_graficos = plotar_graficos(df_final)

    return img_bytes, csv_bytes, figura_graficos
