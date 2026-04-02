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
import matplotlib.gridspec as gridspec

# --- ESTILO DOS BOTÕES (CSS) ---
st.markdown("""
<style>
/* Botões de Ação (Azul) */
.stButton > button {
    background-color: #0072C6;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    border: 1px solid #005A9E;
    padding: 0.5em 1em;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #005A9E;
    color: white;
    border-color: #003B65;
}
/* Botões de Download (Verde) */
.stDownloadButton > button {
    background-color: #1E8A42;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    border: 1px solid #176B34;
    padding: 0.5em 1em;
    transition: background-color 0.3s;
}
.stDownloadButton > button:hover {
    background-color: #176B34;
    color: white;
    border-color: #104A23;
}
</style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE PLOTAGEM E PROCESSAMENTO ---

def plotar_graficos(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 18)) 
    gs = gridspec.GridSpec(3, 2, figure=fig)
    fig.tight_layout(pad=6.0)

    # --- LINHA 1: Gráfico de Trajetória ---
    ax1 = fig.add_subplot(gs[0, :])
    x, y = df['pos_x_um'].to_numpy(), df['pos_y_um'].to_numpy()
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
    ax1.set_title('Gráfico de Trajetória Físico-Espacial', fontsize=16)
    ax1.set_xlabel('Posição X (u.m.)')
    ax1.set_ylabel('Posição Y (u.m.)')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')

    # --- LINHA 2: Gráficos de Velocidade ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['tempo_s'], df['vx_um_s'], label='Velocidade em X', color='green', marker='o', linestyle='--')
    ax2.set_title('Velocidade na Direção X vs. Tempo', fontsize=16)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Velocidade (u.m./s)')
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['tempo_s'], df['vy_um_s'], label='Velocidade em Y', color='orange', marker='o', linestyle='--')
    ax3.set_title('Velocidade na Direção Y vs. Tempo', fontsize=16)
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Velocidade (u.m./s)')
    ax3.legend()

    # --- LINHA 3: Gráficos de Aceleração ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df['tempo_s'], df['ax_um_s2'], label='Aceleração em X', color='purple', marker='^', linestyle='-')
    ax4.set_title('Aceleração na Direção X vs. Tempo', fontsize=16)
    ax4.set_xlabel('Tempo (s)')
    ax4.set_ylabel('Aceleração (u.m./s²)')
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df['tempo_s'], df['ay_um_s2'], label='Aceleração em Y', color='brown', marker='^', linestyle='-')
    ax5.set_title('Aceleração na Direção Y vs. Tempo', fontsize=16)
    ax5.set_xlabel('Tempo (s)')
    ax5.set_ylabel('Aceleração (u.m./s²)')
    ax5.legend()

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

def aplicar_homografia(frame, pts_origem, largura_real, altura_real):
    """Corrige a perspectiva do frame baseado em 4 pontos reais."""
    pts_src = np.array(pts_origem, dtype="float32")
    
    # Determina o tamanho da imagem de destino baseado na proporção real
    largura_dst = 800
    altura_dst = int(largura_dst * (altura_real / largura_real))
    
    pts_dst = np.array([
        [0, 0],
        [largura_dst - 1, 0],
        [largura_dst - 1, altura_dst - 1],
        [0, altura_dst - 1]
    ], dtype="float32")
    
    matriz_perspectiva = cv2.getPerspectiveTransform(pts_src, pts_dst)
    frame_corrigido = cv2.warpPerspective(frame, matriz_perspectiva, (largura_dst, altura_dst))
    return frame_corrigido, matriz_perspectiva, (largura_dst, altura_dst)

def calcular_ajuste_teorico(t, y, grau):
    """Realiza o ajuste polinomial e calcula o R²."""
    coefs = np.polyfit(t, y, grau)
    p = np.poly1d(coefs)
    y_pred = p(t)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    return p, y_pred, r2, coefs

def processar_video(video_bytes, initial_frame, start_frame_idx, bbox_coords_opencv, fator_distancia, scale_factor, origin_coords, status_text_element, window_size=11, poly_order=2, matriz_homografia=None, dimensao_homografia=None):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    # Tracker CSRT é robusto a mudanças de textura e iluminação
    tracker = cv2.TrackerCSRT_create()
    tracker.init(initial_frame, bbox_coords_opencv)

    imagem_estroboscopica = initial_frame.copy()
    altura_frame, largura_frame, _ = initial_frame.shape
    
    # Configuração do Exportador de Vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_video = cv2.VideoWriter(temp_video_out.name, fourcc, fps, (largura_frame, altura_frame))
    
    carimbos_data = []
    posicao_ultimo_carimbo_px = (bbox_coords_opencv[0] + bbox_coords_opencv[2]/2, bbox_coords_opencv[1] + bbox_coords_opencv[3]/2)
    carimbos_data.append([start_frame_idx, posicao_ultimo_carimbo_px[0], posicao_ultimo_carimbo_px[1]])

    contador_frames_processados = 0
    while True:
        frame_atual_idx = start_frame_idx + contador_frames_processados
        if frame_atual_idx >= total_frames: break

        success, frame_atual = cap.read()
        if not success: break
        
        # Se a homografia estiver ativa, corrige a perspectiva do frame do vídeo antes de rastrear
        if matriz_homografia is not None:
            frame_atual = cv2.warpPerspective(frame_atual, matriz_homografia, dimensao_homografia)
        
        status_text_element.text(f"Processando e Rastreando frame {frame_atual_idx}/{total_frames-1}...")
        
        success_track, bbox_atual = tracker.update(frame_atual)
        frame_video_out = frame_atual.copy() # Cópia para o vídeo exportado
        
        if success_track:
            centro_atual_px = (bbox_atual[0] + bbox_atual[2]/2, bbox_atual[1] + bbox_atual[3]/2)
            dist_pixels = np.sqrt((centro_atual_px[0] - posicao_ultimo_carimbo_px[0])**2 + (centro_atual_px[1] - posicao_ultimo_carimbo_px[1])**2)
            
            (x, y, w, h) = [int(v) for v in bbox_atual]
            
            # Desenha o Bounding Box no frame que será exportado para vídeo
            cv2.rectangle(frame_video_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame_video_out, (int(centro_atual_px[0]), int(centro_atual_px[1])), 4, (0, 0, 255), -1)
            
            if dist_pixels * scale_factor >= fator_distancia:
                x_s, y_s, x_e, y_e = max(x, 0), max(y, 0), min(x + w, largura_frame), min(y + h, altura_frame)
                regiao = frame_atual[y_s:y_e, x_s:x_e]
                if regiao.size > 0:
                    imagem_estroboscopica[y_s:y_e, x_s:x_e] = regiao
                carimbos_data.append([frame_atual_idx, centro_atual_px[0], centro_atual_px[1]])
                posicao_ultimo_carimbo_px = centro_atual_px

        # Grava o frame no arquivo de vídeo de saída
        out_video.write(frame_video_out)
        contador_frames_processados += 1
    
    cap.release()
    out_video.release()
    os.remove(video_path)
    
    # Lê os bytes do vídeo gerado
    with open(temp_video_out.name, 'rb') as f:
        video_track_bytes = f.read()
    os.remove(temp_video_out.name)
    
    if len(carimbos_data) < 2: return None
    
    df_carimbos = pd.DataFrame(carimbos_data, columns=['frame', 'pos_x_px', 'pos_y_px'])
    df_carimbos['tempo_s'] = (df_carimbos['frame'] - start_frame_idx) / fps
    df_carimbos['pos_x_um'] = (df_carimbos['pos_x_px'] - origin_coords[0]) * scale_factor
    df_carimbos['pos_y_um'] = -(df_carimbos['pos_y_px'] - origin_coords[1]) * scale_factor

    # Cálculo Cinemático (Savitzky-Golay vs Diferenças Finitas) 
    if len(df_carimbos) > window_size:
        dt = 1.0 / fps
        df_carimbos['pos_x_um'] = savgol_filter(df_carimbos['pos_x_um'], window_length=window_size, polyorder=poly_order, deriv=0)
        df_carimbos['pos_y_um'] = savgol_filter(df_carimbos['pos_y_um'], window_length=window_size, polyorder=poly_order, deriv=0)
        df_carimbos['vx_um_s'] = savgol_filter(df_carimbos['pos_x_um'], window_length=window_size, polyorder=poly_order, deriv=1, delta=dt)
        df_carimbos['vy_um_s'] = savgol_filter(df_carimbos['pos_y_um'], window_length=window_size, polyorder=poly_order, deriv=1, delta=dt)
        df_carimbos['ax_um_s2'] = savgol_filter(df_carimbos['pos_x_um'], window_length=window_size, polyorder=poly_order, deriv=2, delta=dt)
        df_carimbos['ay_um_s2'] = savgol_filter(df_carimbos['pos_y_um'], window_length=window_size, polyorder=poly_order, deriv=2, delta=dt)
    else:
        delta_t = df_carimbos['tempo_s'].diff()
        df_carimbos['vx_um_s'] = df_carimbos['pos_x_um'].diff() / delta_t
        df_carimbos['vy_um_s'] = df_carimbos['pos_y_um'].diff() / delta_t
        df_carimbos['ax_um_s2'] = df_carimbos['vx_um_s'].diff() / delta_t
        df_carimbos['ay_um_s2'] = df_carimbos['vy_um_s'].diff() / delta_t
    
    df_final = df_carimbos.fillna(0)
    status_text_element.success(f"Processamento concluído! {len(df_final)} pontos extraídos.")
    
    _, buffer_img_estrob = cv2.imencode('.PNG', imagem_estroboscopica)
    img_estrob_bytes = BytesIO(buffer_img_estrob).getvalue()
    figura_graficos = plotar_graficos(df_final)
    
    return img_estrob_bytes, df_final, figura_graficos, video_track_bytes

def desenhar_vetores_velocidade(imagem_estroboscopica_original, df_analisado, scale_vetor, max_len_vetor, cor_vetor, espessura_vetor):
    imagem_com_vetores = imagem_estroboscopica_original.copy()
    for i in range(1, len(df_analisado)):
        p_start_px = (int(df_analisado.loc[i, 'pos_x_px']), int(df_analisado.loc[i, 'pos_y_px']))
        vx, vy = df_analisado.loc[i, 'vx_um_s'], df_analisado.loc[i, 'vy_um_s']
        vel_magnitude = np.sqrt(vx**2 + vy**2)
        if not np.isnan(vx) and not np.isnan(vy) and vel_magnitude > 0:
            arrow_length_pixels = min(max_len_vetor, vel_magnitude * scale_vetor)
            direction_x, direction_y = vx / vel_magnitude, vy / vel_magnitude
            p_end_px = (int(p_start_px[0] + direction_x * arrow_length_pixels), int(p_start_px[1] - direction_y * arrow_length_pixels))
            cv2.arrowedLine(imagem_com_vetores, p_start_px, p_end_px, cor_vetor, espessura_vetor, tipLength=0.3)
    _, buffer = cv2.imencode('.PNG', imagem_com_vetores)
    return BytesIO(buffer).getvalue()

# --- INTERFACE DA APLICAÇÃO ---

st.set_page_config(layout="wide", page_title="Análise de Movimento por Vídeo")
st.markdown("# 🔬 Análise de Movimento por Vídeo")
st.markdown("### Uma ferramenta para extrair dados cinemáticos de vídeos com câmera estática.")

if 'step' not in st.session_state: st.session_state.step = "upload"
if 'results' not in st.session_state: st.session_state.results = None

if st.session_state.step == "upload":
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## Passo 1: Upload do Vídeo")
    with col2:
        if st.button("🔄 Analisar novo vídeo"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    video_file = st.file_uploader("Escolha um arquivo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    if video_file:
        st.session_state.video_bytes = video_file.getvalue()
        st.session_state.step = "frame_selection"
        st.session_state.results = None 
        st.rerun()

if st.session_state.step == "frame_selection":
    st.markdown("## Passo 2: Seleção do Frame Inicial")
    st.info("Navegue pelos frames para escolher o momento exato em que a análise deve começar.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(st.session_state.video_bytes)
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if 'current_frame_idx' not in st.session_state: 
        st.session_state.current_frame_idx = 0

    st.markdown("##### Controles de Navegação")
    nav_cols = st.columns([1, 1, 1])

    with nav_cols[0]:
        if st.button("<< Frame Anterior"):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
    with nav_cols[2]:
        if st.button("Próximo Frame >>"):
            st.session_state.current_frame_idx = min(total_frames - 1, st.session_state.current_frame_idx + 1)
    with nav_cols[1]:
        input_cols = st.columns([1, 2, 1])
        with input_cols[1]:
            st.number_input("Ir para o Frame:", min_value=0, max_value=total_frames - 1, step=1, key="current_frame_idx", label_visibility="visible")

    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
    success, frame = cap.read()
    if success: 
        img_cols = st.columns([1, 8, 1])
        with img_cols[1]:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Exibindo Frame: {st.session_state.current_frame_idx} / {total_frames-1}")
        
    if st.button("Confirmar Frame e Iniciar Configuração", type="primary"):
        st.session_state.raw_initial_frame = frame
        st.session_state.start_frame_for_analysis = st.session_state.current_frame_idx
        st.session_state.step = "configuration"
        st.rerun()
    
    cap.release(), os.remove(tfile.name)

if st.session_state.step == "configuration":
    st.markdown("## Passo 3: Configuração e Análise")
    st.info("Ajuste os parâmetros abaixo. O layout responsivo se adapta à sua tela.")
    
    # Controle de Homografia
    usar_homografia = st.checkbox("📐 Ativar Correção de Perspectiva (Avançado)", help="Corrige vídeos gravados em ângulo.")
    
    matriz_H = None
    dim_H = None
    
    # Prevenção de erro: Garante que a imagem está na memória antes de copiar
    if 'raw_initial_frame' not in st.session_state:
        if 'initial_frame' in st.session_state:
            # Puxa da versão antiga se o usuário estiver no meio do reload
            st.session_state.raw_initial_frame = st.session_state.initial_frame
        else:
            # Trava a execução e avisa o usuário caso a memória esteja vazia
            st.warning("Sessão expirada ou página recarregada. Por favor, volte ao Passo 1 e carregue o vídeo novamente.")
            st.stop()
            
    frame_trabalho = st.session_state.raw_initial_frame.copy()
    
    if usar_homografia:
        st.warning("Insira as coordenadas X,Y de 4 pontos que formam um retângulo na vida real (ex: uma folha A4 no fundo).")
        hc1, hc2, hc3, hc4 = st.columns(4)
        hx1 = hc1.number_input("Sup. Esq. X", 0, key='hx1'); hy1 = hc1.number_input("Sup. Esq. Y", 0, key='hy1')
        hx2 = hc2.number_input("Sup. Dir. X", 100, key='hx2'); hy2 = hc2.number_input("Sup. Dir. Y", 0, key='hy2')
        hx3 = hc3.number_input("Inf. Dir. X", 100, key='hx3'); hy3 = hc3.number_input("Inf. Dir. Y", 100, key='hy3')
        hx4 = hc4.number_input("Inf. Esq. X", 0, key='hx4'); hy4 = hc4.number_input("Inf. Esq. Y", 100, key='hy4')
        
        larg_real_H = st.number_input("Largura Real desse Retângulo (u.m.)", value=1.0)
        alt_real_H = st.number_input("Altura Real desse Retângulo (u.m.)", value=1.0)
        
        if st.button("Pré-visualizar Correção"):
            pts_origem = [[hx1, hy1], [hx2, hy2], [hx3, hy3], [hx4, hy4]]
            frame_trabalho, matriz_H, dim_H = aplicar_homografia(st.session_state.raw_initial_frame, pts_origem, larg_real_H, alt_real_H)
            st.session_state.frame_trabalho = frame_trabalho
            st.session_state.matriz_H = matriz_H
            st.session_state.dim_H = dim_H
    
    # Usa o frame corrigido se a homografia foi ativada, senão usa o original
    frame_ativo = st.session_state.get('frame_trabalho', st.session_state.raw_initial_frame)
    frame_com_grade = desenhar_grade_cartesiana(frame_ativo)
    altura_total, _, _ = frame_com_grade.shape
    
    orig_x = st.session_state.get('orig_x', 0); orig_y_usuario = st.session_state.get('orig_y', 0)
    x1 = st.session_state.get('x1', 0); y1_usuario = st.session_state.get('y1', 0)
    x2 = st.session_state.get('x2', 0); y2_usuario = st.session_state.get('y2', 0)
    obj_x = st.session_state.get('obj_x', 0); obj_y_usuario = st.session_state.get('obj_y', 0)
    obj_w = st.session_state.get('obj_w', 50); obj_h = st.session_state.get('obj_h', 50)

    frame_para_preview = frame_com_grade.copy()
    orig_y_opencv_preview = altura_total - orig_y_usuario
    y1_opencv_preview = altura_total - y1_usuario
    y2_opencv_preview = altura_total - y2_usuario
    obj_y_opencv_preview = altura_total - obj_y_usuario - obj_h

    cv2.circle(frame_para_preview, (orig_x, orig_y_opencv_preview), 10, (255, 0, 255), -1)
    cv2.circle(frame_para_preview, (x1, y1_opencv_preview), 5, (0, 255, 255), -1)
    cv2.circle(frame_para_preview, (x2, y2_opencv_preview), 5, (0, 255, 255), -1)
    cv2.line(frame_para_preview, (x1, y1_opencv_preview), (x2, y2_opencv_preview), (0, 255, 255), 2)
    if obj_w > 0 and obj_h > 0: 
        cv2.rectangle(frame_para_preview, (obj_x, obj_y_opencv_preview), (obj_x + obj_w, obj_y_opencv_preview + obj_h), (255, 0, 0), 2)
    
    img_col1, img_col2, img_col3 = st.columns([1, 4, 1])
    with img_col2:
        st.image(cv2.cvtColor(frame_para_preview, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Origem e Calibração")
        orig_x = st.number_input("Origem (0,0) - Eixo X", 0, step=10, key="orig_x")
        orig_y_usuario = st.number_input("Origem (0,0) - Eixo Y", 0, step=10, key="orig_y")
        st.markdown("Espaço Real:")
        p1, p2 = st.columns(2)
        x1 = p1.number_input("Ponto 1 - X", 0, step=10, key="x1")
        y1_usuario = p2.number_input("Ponto 1 - Y", 0, step=10, key="y1")
        x2 = p1.number_input("Ponto 2 - X", 0, step=10, key="x2")
        y2_usuario = p2.number_input("Ponto 2 - Y", 0, step=10, key="y2")
        distancia_real = st.number_input("Distância real (u.m.)", min_value=0.01, format="%.4f", key="dist_real")

    with col2:
        st.markdown("#### 2. Rastreamento do Objeto")
        obj_x = st.number_input("Canto Esquerdo - X", 0, step=10, key="obj_x")
        obj_y_usuario = st.number_input("Canto Inferior - Y", 0, step=10, key="obj_y")
        obj_w = st.number_input("Largura", min_value=10, value=50, step=10, key="obj_w")
        obj_h = st.number_input("Altura", min_value=10, value=50, step=10, key="obj_h")

    with col3:
        st.markdown("#### 3. Algoritmo e Suavização")
        fator_dist = st.slider("Espaçamento de Captura", 0.01, 5.0, 0.5, 0.01)
        st.markdown("**Filtro Savitzky-Golay:**")
        window_size = st.slider("Tamanho da Janela", min_value=5, max_value=51, value=11, step=2)
        poly_order = st.slider("Ordem do Polinômio", min_value=1, max_value=4, value=2)

    # ---------------------------------------------------------
    # PARTE 3: BOTÃO DE AÇÃO PRINCIPAL
    # ---------------------------------------------------------
    st.markdown("---")
    if window_size <= poly_order:
        st.error("Erro: O tamanho da janela do filtro deve ser maior que a ordem do polinômio.")
    else:
        if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
            status_text = st.empty()
            with st.spinner("Extraindo cinemática..."):
                orig_y_opencv = altura_total - orig_y_usuario
                y1_opencv = altura_total - y1_usuario
                y2_opencv = altura_total - y2_usuario
                origin_coords = (orig_x, orig_y_opencv)
                length_pixels = np.sqrt((x2 - x1)**2 + (y2_opencv - y1_opencv)**2)
                
                if length_pixels > 0:
                    scale_factor = distancia_real / length_pixels
                    obj_y_opencv = altura_total - obj_y_usuario - obj_h
                    bbox_opencv = (obj_x, obj_y_opencv, obj_w, obj_h)
                    
                    # --- CORREÇÃO: Criação do cabeçalho CSV ---
                    header_comentarios = (
                        f"# Análise de Movimento - {pd.Timestamp.now()}\n"
                        f"# Frame Inicial: {st.session_state.start_frame_for_analysis}\n"
                        f"# Origem (pixels): {origin_coords}\n"
                        f"# Fator de Escala: {scale_factor:.6f} u.m./pixel\n"
                        f"# --- \n"
                    )
                    st.session_state.csv_header = header_comentarios
                    # ------------------------------------------
                    
                    st.session_state.results = processar_video(
                        st.session_state.video_bytes, frame_ativo, st.session_state.start_frame_for_analysis, 
                        bbox_opencv, fator_dist, scale_factor, origin_coords, status_text, window_size, poly_order,
                        st.session_state.get('matriz_H', None), st.session_state.get('dim_H', None)
                    )
                else: 
                    st.error("A distância da calibração não pode ser zero.")

if st.session_state.results:
    st.markdown("---")
    st.markdown("## ✅ Resultados da Análise Principal")
    img_estrob_bytes, df_final, figura_graficos, video_track_bytes = st.session_state.results

    if img_estrob_bytes:
        with st.expander("📊 Resultados Detalhados (Imagens e Gráficos)", expanded=True):
            st.markdown("### Imagem Estroboscópica e Vídeo de Rastreamento")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.image(img_estrob_bytes, caption="Imagem Composta")
            with res_col2:
                st.download_button("🎬 Baixar Vídeo com Rastreio (.mp4)", video_track_bytes, "video_rastreado.mp4", "video/mp4", use_container_width=True)
                st.download_button("💾 Baixar Imagem Composta (.png)", img_estrob_bytes, "imagem_estroboscopica.png", "image/png", use_container_width=True)

            st.markdown("### Gráficos de Cinemática")
            st.pyplot(figura_graficos)
            
            # --- RESTAURADO: Tabela e Exportação CSV ---
            st.markdown("### Tabela de Dados Brutos e Suavizados")
            st.dataframe(df_final)
            csv_final_string = st.session_state.csv_header + df_final.to_csv(index=False)
            st.download_button("💾 Baixar Tabela de Dados (.csv)", csv_final_string, "dados_analise.csv", "text/csv", use_container_width=True)
            
        with st.expander("📈 Ajuste de Curvas Teóricas (Modelos Físicos)"):
            st.info("Compare os dados reais extraídos pelo vídeo com as equações teóricas da física clássica.")
            aj_col1, aj_col2 = st.columns(2)
            eixo_ajuste = aj_col1.selectbox("Eixo de Análise", ["Posição Y", "Posição X"])
            modelo_ajuste = aj_col2.selectbox("Modelo Físico", ["Linear (Mov. Uniforme)", "Quadrático (Mov. Uniformemente Variado)"])
            
            y_data = df_final['pos_y_um'] if eixo_ajuste == "Posição Y" else df_final['pos_x_um']
            t_data = df_final['tempo_s']
            grau = 1 if "Linear" in modelo_ajuste else 2
            
            p_func, y_pred, r2, coefs = calcular_ajuste_teorico(t_data, y_data, grau)
            
            st.markdown(f"**Coeficiente de Determinação ($R^2$):** `{r2:.4f}` (Quanto mais próximo de 1, mais perfeito é o experimento)")
            if grau == 1:
                st.markdown(f"**Equação da Reta:** $S(t) = {coefs[0]:.4f}t + ({coefs[1]:.4f})$")
                st.markdown(f"*Velocidade Média Constante calculada:* `{coefs[0]:.4f} u.m./s`")
            else:
                st.markdown(f"**Equação da Parábola:** $S(t) = {coefs[0]:.4f}t^2 + {coefs[1]:.4f}t + {coefs[2]:.4f}$")
                st.markdown(f"*Aceleração Constante calculada ($2a$):* `{coefs[0]*2:.4f} u.m./s²`")
            
            fig_ajuste, ax_aj = plt.subplots(figsize=(10, 4))
            ax_aj.scatter(t_data, y_data, color='blue', alpha=0.5, label='Dados Reais (Vídeo)')
            ax_aj.plot(t_data, y_pred, color='red', linewidth=2, label=f'Ajuste Teórico {modelo_ajuste}')
            ax_aj.set_xlabel("Tempo (s)"); ax_aj.set_ylabel(eixo_ajuste)
            ax_aj.legend(); ax_aj.grid(True)
            st.pyplot(fig_ajuste)

        # --- RESTAURADO: Vetores de Velocidade ---
        with st.expander("🏹 Análise Adicional: Vetores de Velocidade"):
            st.info("Ajuste os parâmetros e clique para gerar uma imagem com os vetores direcionais.")
            cores_bgr = {"Vermelho": (0, 0, 255), "Azul": (255, 0, 0), "Amarelo": (0, 255, 255), "Ciano": (255, 255, 0), "Magenta": (255, 0, 255), "Verde": (0, 255, 0), "Branco": (255, 255, 255), "Laranja": (0, 165, 255)}
            
            vec_col1, vec_col2 = st.columns(2)
            cor_nome = vec_col1.selectbox("Cor do Vetor", options=list(cores_bgr.keys()))
            espessura_vetor = vec_col2.slider("Espessura do Vetor (px)", 1, 5, 2)
            scale_vetor_col, max_len_col = st.columns(2)
            scale_vetor = scale_vetor_col.slider("Escala do Vetor", 1, 200, 50, help="Multiplicador para o comprimento dos vetores.")
            max_len_vetor = max_len_col.slider("Comprimento Máximo (px)", 10, 200, 100, help="Limite visual do tamanho da seta.")
            
            if st.button("Gerar / Atualizar Imagem com Vetores", use_container_width=True):
                imagem_original = cv2.imdecode(np.frombuffer(img_estrob_bytes, np.uint8), 1)
                img_vetores_bytes = desenhar_vetores_velocidade(imagem_original, df_final, scale_vetor, max_len_vetor, cores_bgr[cor_nome], espessura_vetor)
                st.session_state.img_vetores = img_vetores_bytes
            
            if 'img_vetores' in st.session_state and st.session_state.img_vetores:
                st.image(st.session_state.img_vetores, caption="Imagem com Vetores de Velocidade")
                st.download_button("💾 Baixar Imagem com Vetores (.png)", st.session_state.img_vetores, "imagem_com_vetores.png", "image/png", use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Analisar outro vídeo", key="btn_reset_final"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

# --- Rodapé Informativo ---
st.markdown("---")
footer_expander = st.expander("💡 Informações Adicionais (Funcionalidades e Dicas)", expanded=False)
with footer_expander:
    st.markdown("#### Funcionalidades e Inovações Técnicas")
    st.markdown("""
    - **Seleção de Frame Inicial:** Navegue pelo vídeo para definir o ponto exato de início da análise.
    - **Calibração de Espaço:** Defina uma origem (0,0) e uma escala de referência (u.m./pixel) para obter dados em unidades de medida reais.
    - **Rastreamento de Objeto:** Acompanha o objeto selecionado ao longo do vídeo.
    - **Análise Cinemática:** Calcula e exibe dados de posição e velocidade (componentes X/Y).
    - **Visualização de Dados:** Gera uma imagem estroboscópica, gráficos de trajetória, velocidade e aceleração (validação de forças), e uma imagem opcional com vetores de velocidade.
    - **Exportação de Resultados:** Permite o download da imagem estroboscópica e da tabela de dados completa em formato CSV, incluindo os parâmetros da análise.
    - **Rastreio CSRT:** Utiliza o algoritmo *Discriminative Correlation Filter* para rastreamento robusto, resistente a falhas de textura e iluminação.
    - **Homografia (Correção de Perspectiva):** Permite corrigir vídeos gravados fora de esquadro antes do rastreamento.
    - **Exportação de Vídeo:** Comprova a validade do experimento permitindo o download do vídeo com o *bounding box* anexado a cada quadro.
    - **Ajuste de Curvas (Trendlines):** Permite extrair a velocidade ou aceleração constante teórica validando as leis da cinemática clássica (com cálculo rigoroso de $R^2$).
    - **Suavização Savitzky-Golay:** Derivação polinomial avançada acoplada que limpa ruídos de quantização de pixels e devolve campos de aceleração fisicamente coerentes.
    """)
    st.markdown("#### Dicas para Melhores Resultados (O Segredo está na Gravação!)")
    st.markdown("""
    - **Câmera Estritamente Estática:** Para um resultado preciso, é fundamental que o vídeo tenha sido gravado com a **câmera completamente parada** (use um tripé). Qualquer vibração compromete a extração matemática da velocidade e aceleração.
    - **Dados Precisos (Visão Lateral):** A gravação deve ser realizada em um plano estritamente perpendicular ao do movimento do objeto (visão 2D, de lado). Evite gravar o objeto se movendo em direção à lente para não gerar distorção de perspectiva.
    - **A Importância do FPS (Taxa de Quadros):** O FPS é a sua "régua de tempo". Para fenômenos muito rápidos (como colisões ou a inversão de movimento de um pêndulo), grave o vídeo com um FPS alto (60 ou 120 FPS). Isso fornece mais amostras de dados ao algoritmo e evita que o evento ocorra "no ponto cego" entre dois quadros.
    - **Cuidado com a Taxa de Quadros Variável (VFR):** Câmeras nativas de smartphones modernos frequentemente alteram a taxa de quadros durante o vídeo para lidar com a iluminação, gravando em VFR. O algoritmo assume uma taxa constante (CFR). Se o celular variar o FPS na gravação, você verá oscilações ou "lombadas" artificiais nos gráficos de aceleração. Se possível, utilize aplicativos de câmera de terceiros (como o *Open Camera*) para "travar" o FPS da gravação em um valor fixo.
    - **Iluminação e Contraste:** O rastreamento de cor exige que o objeto se destaque do fundo e não passe por sombras profundas durante o movimento.
    """)
