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
from streamlit_image_coordinates import streamlit_image_coordinates

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

def processar_video(video_bytes, initial_frame, start_frame_idx, end_frame_idx, bbox_coords_opencv, fator_distancia, scale_factor, origin_coords, status_text_element, window_size=11, poly_order=2, matriz_homografia=None, dimensao_homografia=None, fps_override=0.0):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_bytes)
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    # --- LÓGICA DE TEMPO BLINDADA ---
    fps_metadados = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = fps_override if fps_override > 0 else fps_metadados
    # --------------------------------
    
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
    # Inicializamos a posição de referência, mas NÃO adicionamos à lista ainda (Resolve o bug do ponto duplo)
    posicao_ultimo_carimbo_px = (bbox_coords_opencv[0] + bbox_coords_opencv[2]/2, bbox_coords_opencv[1] + bbox_coords_opencv[3]/2)

    contador_frames_processados = 0
    while True:
        frame_atual_idx = start_frame_idx + contador_frames_processados
        
        # --- O LAÇO PARA NO FRAME FINAL ---
        if frame_atual_idx > end_frame_idx or frame_atual_idx >= total_frames: 
            break

        success, frame_atual = cap.read()
        if not success: break
        
        # Se a homografia estiver ativa, corrige a perspectiva do frame do vídeo antes de rastrear
        if matriz_homografia is not None:
            frame_atual = cv2.warpPerspective(frame_atual, matriz_homografia, dimensao_homografia)
        
        status_text_element.text(f"Processando e Rastreando frame {frame_atual_idx}/{end_frame_idx}...")
        
        success_track, bbox_atual = tracker.update(frame_atual)
        frame_video_out = frame_atual.copy() # Cópia para o vídeo exportado
        
        if success_track:
            centro_atual_px = (bbox_atual[0] + bbox_atual[2]/2, bbox_atual[1] + bbox_atual[3]/2)
            dist_pixels = np.sqrt((centro_atual_px[0] - posicao_ultimo_carimbo_px[0])**2 + (centro_atual_px[1] - posicao_ultimo_carimbo_px[1])**2)
            
            (x, y, w, h) = [int(v) for v in bbox_atual]
            
            # Desenha o Bounding Box no frame que será exportado para vídeo
            cv2.rectangle(frame_video_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame_video_out, (int(centro_atual_px[0]), int(centro_atual_px[1])), 4, (0, 0, 255), -1)
            
            # Salva o dado matemático de TODO frame
            carimbos_data.append([frame_atual_idx, centro_atual_px[0], centro_atual_px[1]])
            
            is_stamp = False
            
            # O primeiro frame (contador == 0) SEMPRE será um carimbo na imagem composta
            if contador_frames_processados == 0 or (dist_pixels * scale_factor >= fator_distancia):
                is_stamp = True 
                x_s, y_s, x_e, y_e = max(x, 0), max(y, 0), min(x + w, largura_frame), min(y + h, altura_frame)
                regiao = frame_atual[y_s:y_e, x_s:x_e]
                
                if regiao.size > 0:
                    imagem_estroboscopica[y_s:y_e, x_s:x_e] = regiao
                
                posicao_ultimo_carimbo_px = centro_atual_px

            carimbos_data[-1].append(is_stamp) # Adiciona a flag no último elemento

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
    
    df_carimbos = pd.DataFrame(carimbos_data, columns=['frame', 'pos_x_px', 'pos_y_px', 'is_stamp'])
    df_carimbos['tempo_s'] = (df_carimbos['frame'] - start_frame_idx) / fps
    df_carimbos['pos_x_um'] = (df_carimbos['pos_x_px'] - origin_coords[0]) * scale_factor
    df_carimbos['pos_y_um'] = -(df_carimbos['pos_y_px'] - origin_coords[1]) * scale_factor

    # Cálculo Cinemático (Savitzky-Golay vs Diferenças Finitas) 
    if len(df_carimbos) > window_size:
        dt = 1.0 / fps
        
        # --- A CORREÇÃO DE OURO: BLINDAGEM DOS DADOS BRUTOS ---
        # Extraímos os arrays originais (com ruído do vídeo) para alimentar o filtro.
        pos_x_raw = df_carimbos['pos_x_um'].to_numpy()
        pos_y_raw = df_carimbos['pos_y_um'].to_numpy()
        
        # 1. Velocidade (Primeira Derivada DIRETO do dado bruto)
        df_carimbos['vx_um_s'] = savgol_filter(pos_x_raw, window_length=window_size, polyorder=poly_order, deriv=1, delta=dt)
        df_carimbos['vy_um_s'] = savgol_filter(pos_y_raw, window_length=window_size, polyorder=poly_order, deriv=1, delta=dt)
        
        # 2. Aceleração (Segunda Derivada DIRETO do dado bruto)
        # Proteção matemática: A 2ª derivada requer um polinômio de ordem >= 2. 
        # Se o usuário escolheu 1, forçamos o cálculo da aceleração com 2 para não quebrar a física.
        ordem_acc = poly_order if poly_order >= 2 else 2
        df_carimbos['ax_um_s2'] = savgol_filter(pos_x_raw, window_length=window_size, polyorder=ordem_acc, deriv=2, delta=dt)
        df_carimbos['ay_um_s2'] = savgol_filter(pos_y_raw, window_length=window_size, polyorder=ordem_acc, deriv=2, delta=dt)
        
        # 3. Posição (Suavização final - Derivada 0)
        # Fazemos isso por último para não poluir os cálculos derivados acima!
        df_carimbos['pos_x_um'] = savgol_filter(pos_x_raw, window_length=window_size, polyorder=poly_order, deriv=0)
        df_carimbos['pos_y_um'] = savgol_filter(pos_y_raw, window_length=window_size, polyorder=poly_order, deriv=0)
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
    
    # iterrows() garante que a função funcione mesmo se a tabela for filtrada
    for index, row in df_analisado.iterrows():
        p_start_px = (int(row['pos_x_px']), int(row['pos_y_px']))
        vx, vy = row['vx_um_s'], row['vy_um_s']
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
    st.markdown("## Passo 2: Seleção do Intervalo de Análise")
    st.info("Defina o momento exato de início e fim do movimento. Use a barra para navegar e os botões para marcar os limites.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(st.session_state.video_bytes)
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if 'preview_idx' not in st.session_state: st.session_state.preview_idx = 0
    if 'start_idx' not in st.session_state: st.session_state.start_idx = 0
    if 'end_idx' not in st.session_state: st.session_state.end_idx = total_frames - 1

    # --- BARRA DE NAVEGAÇÃO RÁPIDA ---
    slider_prev = st.slider("Linha do Tempo do Vídeo (Arraste para visualizar)", 0, total_frames - 1, st.session_state.preview_idx)
    if slider_prev != st.session_state.preview_idx:
        st.session_state.preview_idx = slider_prev
        st.rerun()

    nav_cols = st.columns([1, 1, 1, 1])
    if nav_cols[0].button("<< -5 Frames", use_container_width=True): st.session_state.preview_idx = max(0, st.session_state.preview_idx - 5); st.rerun()
    if nav_cols[1].button("< Anterior", use_container_width=True): st.session_state.preview_idx = max(0, st.session_state.preview_idx - 1); st.rerun()
    if nav_cols[2].button("Próximo >", use_container_width=True): st.session_state.preview_idx = min(total_frames-1, st.session_state.preview_idx + 1); st.rerun()
    if nav_cols[3].button("+5 Frames >>", use_container_width=True): st.session_state.preview_idx = min(total_frames-1, st.session_state.preview_idx + 5); st.rerun()

    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.preview_idx)
    success, frame = cap.read()
    if success:
        img_cols = st.columns([1, 8, 1])
        with img_cols[1]:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Visualizando Frame: {st.session_state.preview_idx} / {total_frames-1}")

    # --- BOTÕES DE MARCAÇÃO DE CORTE ---
    st.markdown("### ✂️ Definir Limites de Corte")
    lim_col1, lim_col2 = st.columns(2)
    with lim_col1:
        st.markdown(f"**Frame Inicial:** `{st.session_state.start_idx}`")
        if st.button("📍 Marcar Preview como INICIAL", use_container_width=True):
            st.session_state.start_idx = st.session_state.preview_idx
            if st.session_state.start_idx > st.session_state.end_idx:
                st.session_state.end_idx = st.session_state.start_idx # Proteção contra limites invertidos
            st.rerun()
    with lim_col2:
        st.markdown(f"**Frame Final:** `{st.session_state.end_idx}`")
        if st.button("🛑 Marcar Preview como FINAL", use_container_width=True):
            st.session_state.end_idx = st.session_state.preview_idx
            if st.session_state.end_idx < st.session_state.start_idx:
                st.session_state.start_idx = st.session_state.end_idx
            st.rerun()

    st.markdown("---")
    if st.button("✅ Confirmar Intervalo e Iniciar Configuração", type="primary", use_container_width=True):
        # A imagem base que vai para configuração TEM que ser o Frame Inicial selecionado!
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.start_idx)
        _, frame_inicial_real = cap.read()
        
        st.session_state.raw_initial_frame = frame_inicial_real
        st.session_state.start_frame_for_analysis = st.session_state.start_idx
        st.session_state.end_frame_for_analysis = st.session_state.end_idx
        st.session_state.step = "configuration"
        st.rerun()

    cap.release(), os.remove(tfile.name)

if st.session_state.step == "configuration":
    st.markdown("## Passo 3: Configuração e Análise")
    st.info("Ajuste os parâmetros abaixo. O layout responsivo se adapta à sua tela.")

    # 1. INICIALIZAÇÃO BLINDADA DO SESSION STATE
    chaves_padrao = {
        'orig_x': 0, 'orig_y': 0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0,
        'obj_x': 0, 'obj_y': 0, 'obj_w': 50, 'obj_h': 50,
        # Valores padrão para a perspectiva formam um quadrado visível no canto
        'hx1': 100, 'hy1': 100, 'hx2': 300, 'hy2': 100,
        'hx3': 300, 'hy3': 300, 'hx4': 100, 'hy4': 300
    }
    for k, v in chaves_padrao.items():
        if k not in st.session_state:
            st.session_state[k] = int(v)

    usar_homografia = st.checkbox("📐 Ativar Correção de Perspectiva (Avançado)", help="Corrige vídeos gravados em ângulo.")

    # 2. SELETOR DE FERRAMENTA
    st.markdown("### 🖱️ Calibração Interativa")
    st.info("Selecione um ponto abaixo e clique na imagem para definir sua coordenada.")

    opcoes_ferramenta = ["Nenhum (Apenas Visualizar)", "📍 Origem (0,0)", "📏 Calibração: Ponto 1", "📏 Calibração: Ponto 2", "📦 Objeto: Canto Esquerdo/Inferior"]
    if usar_homografia:
        opcoes_ferramenta.extend(["📐 Perspectiva: Sup. Esq. (1)", "📐 Perspectiva: Sup. Dir. (2)", "📐 Perspectiva: Inf. Dir. (3)", "📐 Perspectiva: Inf. Esq. (4)"])

    ferramenta_ativa = st.radio("Selecione o ponto para marcar no clique:", opcoes_ferramenta, horizontal=True)

    # 3. DETERMINA QUAL IMAGEM EXIBIR
    # Se o usuário está ajustando a perspectiva, ele DEVE ver a imagem original torta.
    if "Perspectiva" in ferramenta_ativa:
        frame_ativo = st.session_state.raw_initial_frame.copy()
        st.warning("⚠️ Exibindo a imagem original sem correção para você marcar os cantos da perspectiva.")
    else:
        frame_ativo = st.session_state.get('frame_trabalho', st.session_state.raw_initial_frame).copy()

    frame_com_grade = desenhar_grade_cartesiana(frame_ativo)
    altura_total, largura_total, _ = frame_com_grade.shape

    # 4. LÓGICA DE DESENHO INTELIGENTE E SINCRONIZADA
    frame_para_preview = frame_com_grade.copy()

    # Desenho da Perspectiva (Laranja)
    if usar_homografia:
        pt1 = (int(st.session_state.hx1), int(st.session_state.hy1))
        pt2 = (int(st.session_state.hx2), int(st.session_state.hy2))
        pt3 = (int(st.session_state.hx3), int(st.session_state.hy3))
        pt4 = (int(st.session_state.hx4), int(st.session_state.hy4))
        
        cv2.polylines(frame_para_preview, [np.array([pt1, pt2, pt3, pt4], dtype=np.int32)], isClosed=True, color=(0, 165, 255), thickness=2)
        for i, pt in enumerate([pt1, pt2, pt3, pt4]):
            cv2.circle(frame_para_preview, pt, 6, (0, 165, 255), -1)
            cv2.putText(frame_para_preview, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Desenho dos Componentes da Cinemática
    # Só desenha se NÃO estivermos ajustando a perspectiva (para evitar visualização quebrada na troca de imagens)
    if "Perspectiva" not in ferramenta_ativa:
        oy_cv = int(altura_total - st.session_state.orig_y)
        y1_cv = int(altura_total - st.session_state.y1)
        y2_cv = int(altura_total - st.session_state.y2)
        obj_y_cv = int(altura_total - st.session_state.obj_y - st.session_state.obj_h)

        cv2.circle(frame_para_preview, (int(st.session_state.orig_x), oy_cv), 10, (255, 0, 255), -1)
        cv2.putText(frame_para_preview, "(0,0)", (int(st.session_state.orig_x) + 15, oy_cv), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if (st.session_state.x1 != 0 or st.session_state.y1 != 0) or "Ponto 1" in ferramenta_ativa:
            cv2.circle(frame_para_preview, (int(st.session_state.x1), y1_cv), 5, (0, 255, 255), -1)
        if (st.session_state.x2 != 0 or st.session_state.y2 != 0) or "Ponto 2" in ferramenta_ativa:
            cv2.circle(frame_para_preview, (int(st.session_state.x2), y2_cv), 5, (0, 255, 255), -1)
        if (st.session_state.x1 != 0 or st.session_state.x2 != 0):
            cv2.line(frame_para_preview, (int(st.session_state.x1), y1_cv), (int(st.session_state.x2), y2_cv), (0, 255, 255), 2)
        if (st.session_state.obj_x != 0 or st.session_state.obj_y != 0) or "Objeto" in ferramenta_ativa:
            cv2.rectangle(frame_para_preview, (int(st.session_state.obj_x), obj_y_cv), (int(st.session_state.obj_x + st.session_state.obj_w), int(obj_y_cv + st.session_state.obj_h)), (255, 0, 0), 2)

    # 5. RENDERIZAÇÃO DA IMAGEM E CAPTURA DE CLIQUE
    img_col1, img_col2, img_col3 = st.columns([1, 5, 1])
    with img_col2:
        larg_orig = frame_para_preview.shape[1]
        alt_orig = frame_para_preview.shape[0]

        LARGURA_TELA = 800
        escala = larg_orig / LARGURA_TELA if larg_orig > LARGURA_TELA else 1.0

        if escala > 1.0:
            frame_exibicao = cv2.resize(frame_para_preview, (LARGURA_TELA, int(alt_orig / escala)))
        else:
            frame_exibicao = frame_para_preview

        imagem_rgb = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2RGB)
        value = streamlit_image_coordinates(imagem_rgb, key="image_click")

        if value is not None:
            if st.session_state.get('last_click') != value:
                st.session_state.last_click = value

                x_click = int(value["x"] * escala)
                y_click = int(value["y"] * escala)
                y_inv_click = int(altura_total - y_click)

                if ferramenta_ativa == "📍 Origem (0,0)":
                    st.session_state.orig_x = x_click; st.session_state.orig_y = y_inv_click; st.rerun()
                elif ferramenta_ativa == "📏 Calibração: Ponto 1":
                    st.session_state.x1 = x_click; st.session_state.y1 = y_inv_click; st.rerun()
                elif ferramenta_ativa == "📏 Calibração: Ponto 2":
                    st.session_state.x2 = x_click; st.session_state.y2 = y_inv_click; st.rerun()
                elif ferramenta_ativa == "📦 Objeto: Canto Esquerdo/Inferior":
                    st.session_state.obj_x = x_click; st.session_state.obj_y = y_inv_click; st.rerun()
                elif ferramenta_ativa == "📐 Perspectiva: Sup. Esq. (1)":
                    st.session_state.hx1 = x_click; st.session_state.hy1 = y_click; st.rerun()
                elif ferramenta_ativa == "📐 Perspectiva: Sup. Dir. (2)":
                    st.session_state.hx2 = x_click; st.session_state.hy2 = y_click; st.rerun()
                elif ferramenta_ativa == "📐 Perspectiva: Inf. Dir. (3)":
                    st.session_state.hx3 = x_click; st.session_state.hy3 = y_click; st.rerun()
                elif ferramenta_ativa == "📐 Perspectiva: Inf. Esq. (4)":
                    st.session_state.hx4 = x_click; st.session_state.hy4 = y_click; st.rerun()

        _, buffer_preview = cv2.imencode('.PNG', frame_para_preview)
        preview_bytes = BytesIO(buffer_preview).getvalue()
        st.download_button("💾 Baixar Imagem de Configuração", preview_bytes, "imagem_configuracao.png", "image/png", use_container_width=True)

    st.markdown("---")

    # 6. INPUTS BLINDADOS E BOTÕES
    if usar_homografia:
        with st.expander("🛠️ Ajuste Manual da Homografia e Execução", expanded=True):
            st.info("Você pode fazer o ajuste fino dos 4 cantos pelos números abaixo. Após configurar, clique em aplicar.")
            hc1, hc2, hc3, hc4 = st.columns(4)
            nhx1 = hc1.number_input("Sup. Esq. X (1)", value=int(st.session_state.hx1), step=10)
            if nhx1 != st.session_state.hx1: st.session_state.hx1 = nhx1; st.rerun()
            nhy1 = hc1.number_input("Sup. Esq. Y (1)", value=int(st.session_state.hy1), step=10)
            if nhy1 != st.session_state.hy1: st.session_state.hy1 = nhy1; st.rerun()

            nhx2 = hc2.number_input("Sup. Dir. X (2)", value=int(st.session_state.hx2), step=10)
            if nhx2 != st.session_state.hx2: st.session_state.hx2 = nhx2; st.rerun()
            nhy2 = hc2.number_input("Sup. Dir. Y (2)", value=int(st.session_state.hy2), step=10)
            if nhy2 != st.session_state.hy2: st.session_state.hy2 = nhy2; st.rerun()

            nhx3 = hc3.number_input("Inf. Dir. X (3)", value=int(st.session_state.hx3), step=10)
            if nhx3 != st.session_state.hx3: st.session_state.hx3 = nhx3; st.rerun()
            nhy3 = hc3.number_input("Inf. Dir. Y (3)", value=int(st.session_state.hy3), step=10)
            if nhy3 != st.session_state.hy3: st.session_state.hy3 = nhy3; st.rerun()

            nhx4 = hc4.number_input("Inf. Esq. X (4)", value=int(st.session_state.hx4), step=10)
            if nhx4 != st.session_state.hx4: st.session_state.hx4 = nhx4; st.rerun()
            nhy4 = hc4.number_input("Inf. Esq. Y (4)", value=int(st.session_state.hy4), step=10)
            if nhy4 != st.session_state.hy4: st.session_state.hy4 = nhy4; st.rerun()

            hc_w, hc_h = st.columns(2)
            larg_real_H = hc_w.number_input("Largura Real do Retângulo (u.m.)", value=1.0)
            alt_real_H = hc_h.number_input("Altura Real do Retângulo (u.m.)", value=1.0)

            if st.button("🔄 Aplicar Correção de Perspectiva", use_container_width=True):
                pts_origem = [[st.session_state.hx1, st.session_state.hy1], [st.session_state.hx2, st.session_state.hy2], [st.session_state.hx3, st.session_state.hy3], [st.session_state.hx4, st.session_state.hy4]]
                frame_trabalho, matriz_H, dim_H = aplicar_homografia(st.session_state.raw_initial_frame, pts_origem, larg_real_H, alt_real_H)
                st.session_state.frame_trabalho = frame_trabalho
                st.session_state.matriz_H = matriz_H
                st.session_state.dim_H = dim_H
                # Força a interface a voltar para "Nenhum" para exibir o resultado desamassado
                st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Origem e Calibração")
        nx = st.number_input("Origem (0,0) - Eixo X", value=int(st.session_state.orig_x), step=10)
        if nx != st.session_state.orig_x: st.session_state.orig_x = nx; st.rerun()
        
        ny = st.number_input("Origem (0,0) - Eixo Y", value=int(st.session_state.orig_y), step=10)
        if ny != st.session_state.orig_y: st.session_state.orig_y = ny; st.rerun()
        
        st.markdown("Espaço Real:")
        p1, p2 = st.columns(2)
        nx1 = p1.number_input("Ponto 1 - X", value=int(st.session_state.x1), step=10)
        if nx1 != st.session_state.x1: st.session_state.x1 = nx1; st.rerun()
        
        ny1 = p2.number_input("Ponto 1 - Y", value=int(st.session_state.y1), step=10)
        if ny1 != st.session_state.y1: st.session_state.y1 = ny1; st.rerun()
        
        nx2 = p1.number_input("Ponto 2 - X", value=int(st.session_state.x2), step=10)
        if nx2 != st.session_state.x2: st.session_state.x2 = nx2; st.rerun()
        
        ny2 = p2.number_input("Ponto 2 - Y", value=int(st.session_state.y2), step=10)
        if ny2 != st.session_state.y2: st.session_state.y2 = ny2; st.rerun()
        
        distancia_real = st.number_input("Distância real (u.m.)", min_value=0.01, value=1.0, format="%.4f", key="dist_real")

    with col2:
        st.markdown("#### 2. Rastreamento do Objeto")
        nox = st.number_input("Canto Esq. - X", value=int(st.session_state.obj_x), step=10)
        if nox != st.session_state.obj_x: st.session_state.obj_x = nox; st.rerun()
        
        noy = st.number_input("Canto Inf. - Y", value=int(st.session_state.obj_y), step=10)
        if noy != st.session_state.obj_y: st.session_state.obj_y = noy; st.rerun()
        
        now = st.number_input("Largura", value=int(st.session_state.obj_w), step=10)
        if now != st.session_state.obj_w: st.session_state.obj_w = now; st.rerun()
        
        noh = st.number_input("Altura", value=int(st.session_state.obj_h), step=10)
        if noh != st.session_state.obj_h: st.session_state.obj_h = noh; st.rerun()

    with col3:
        st.markdown("#### 3. Algoritmo e Suavização")
        fator_dist = st.slider("Espaçamento de Captura", 0.01, 5.0, 0.5, 0.01)
        fps_manual = st.number_input("FPS Manual (Opcional)", min_value=0.0, value=0.0, help="Deixe em 0.0 para usar o FPS nativo do arquivo de vídeo.")
        st.markdown("**Filtro Savitzky-Golay:**")
        window_size = st.slider("Tamanho da Janela", min_value=5, max_value=51, value=11, step=2)
        poly_order = st.slider("Ordem do Polinômio", min_value=1, max_value=4, value=2)
   
    st.markdown("---")
    
    # 6. INÍCIO DA ANÁLISE COM VARIÁVEIS BLINDADAS
    if window_size <= poly_order:
        st.error("Erro: O tamanho da janela do filtro deve ser maior que a ordem do polinômio.")
    else:
        if st.button("🚀 Iniciar Análise", type="primary", use_container_width=True):
            status_text = st.empty()
            with st.spinner("Extraindo cinemática..."):
                oy_cv = int(altura_total - st.session_state.orig_y)
                y1_cv = int(altura_total - st.session_state.y1)
                y2_cv = int(altura_total - st.session_state.y2)
                origin_coords = (int(st.session_state.orig_x), oy_cv)
                
                length_pixels = np.sqrt((st.session_state.x2 - st.session_state.x1)**2 + (y2_cv - y1_cv)**2)
                
                if length_pixels > 0:
                    scale_factor = distancia_real / length_pixels
                    obj_y_cv = int(altura_total - st.session_state.obj_y - st.session_state.obj_h)
                    bbox_opencv = (int(st.session_state.obj_x), obj_y_cv, int(st.session_state.obj_w), int(st.session_state.obj_h))
             
                    header_comentarios = (
                        f"# Análise de Movimento - {pd.Timestamp.now()}\n"
                        f"# Frame Inicial: {st.session_state.start_frame_for_analysis}\n"
                        f"# Frame Final: {st.session_state.end_frame_for_analysis}\n"
                        f"# Origem (pixels): {origin_coords}\n"
                        f"# Fator de Escala: {scale_factor:.6f} u.m./pixel\n"
                        f"# --- \n"
                    )
                    st.session_state.csv_header = header_comentarios
                    
                    st.session_state.results = processar_video(
                        st.session_state.video_bytes, frame_ativo, 
                        st.session_state.start_frame_for_analysis, 
                        st.session_state.end_frame_for_analysis, 
                        bbox_opencv, fator_dist, scale_factor, origin_coords, status_text, window_size, poly_order,
                        st.session_state.get('matriz_H', None), st.session_state.get('dim_H', None),
                        fps_override=fps_manual
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
            
        with st.expander("📈 Ajuste de Curvas Teóricas (Modelos Físicos)", expanded=True):          
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
        with st.expander("🏹 Análise Adicional: Vetores de Velocidade", expanded=True):            
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
                
                # --- NOVO: Filtra a tabela para usar apenas os pontos carimbados ---
                df_apenas_carimbos = df_final[df_final['is_stamp'] == True]
                
                img_vetores_bytes = desenhar_vetores_velocidade(imagem_original, df_apenas_carimbos, scale_vetor, max_len_vetor, cores_bgr[cor_nome], espessura_vetor)
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
