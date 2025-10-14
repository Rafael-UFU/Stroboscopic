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

# --- FUNÇÕES DE PLOTAGEM E PROCESSAMENTO ---

def plotar_graficos(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.tight_layout(pad=5.0)

    # Gráfico 1: Trajetória
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
    ax1.set_title('Gráfico de Trajetória', fontsize=16)
    ax1.set_xlabel('Posição X (u.m.)')
    ax1.set_ylabel('Posição Y (u.m.)')
    ax1.legend()
    ax1.set_aspect('equal', adjustable='box')

    # Gráfico 2: Velocidade
    ax2.plot(df['tempo_s'], df['velocidade_um_s'], label='Velocidade', color='green')
    ax2.set_title('Magnitude da Velocidade vs. Tempo', fontsize=16)
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Velocidade (u.m./s)')
    ax2.legend()

    # Gráfico 3: Aceleração
    ax3.plot(df['tempo_s'], df['aceleracao_um_s2'], label='Aceleração', color='purple')
    ax3.set_title('Magnitude da Aceleração vs. Tempo', fontsize=16)
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Aceleração (u.m./s²)')
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
    
    df['pos_x_um'] = (df['pos_x_px'] - origin_coords[0]) * scale_factor
    df['pos_y_um'] = -(df['pos_y_px'] - origin_coords[1]) * scale_factor
    
    df['velocidade_um_s'] = np.sqrt(df['pos_x_um'].diff()**2 + df['pos_y_um'].diff()**2) / df['tempo_s'].diff()
    window_len = min(51, len(df) - 2 if len(df) % 2 == 0 else len(df) - 1)
    if window_len > 3:
        df['vel_suavizada'] = savgol_filter(df['velocidade_um_s'].fillna(0), window_len, 3)
    else:
        df['vel_suavizada'] = df['velocidade_um_s']
    df['aceleracao_um_s2'] = df['vel_suavizada'].diff() / df['tempo_s'].diff()
    
    df_final = df[['frame', 'tempo_s', 'pos_x_um', 'pos_y_um', 'velocidade_um_s', 'aceleracao_um_s2']].copy().fillna(0)

    status_text_element.success(f"Processamento concluído!")
    
    csv_bytes = df_final.to_csv(index=False).encode('utf-8')
    _, buffer = cv2.imencode('.PNG', imagem_estroboscopica)
    img_bytes = BytesIO(buffer).getvalue()
    figura_graficos = plotar_graficos(df_final)

    return img_bytes, csv_bytes, figura_graficos

# --- INTERFACE DA APLICAÇÃO ---

st.set_page_config(layout="wide", page_title="Análise de Movimento por Vídeo")
st.markdown("# 🔬 Análise de Movimento por Vídeo")
st.markdown("### Uma ferramenta para extrair dados cinemáticos de vídeos com câmera estática.")

# Inicializa o estado da sessão
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'results' not in st.session_state:
    st.session_state.results = None

# --- PASSO 0: UPLOAD ---
if st.session_state.step == "upload":
    st.markdown("## Passo 1: Upload do Vídeo")
    video_file = st.file_uploader("Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    
    if video_file:
        st.session_state.video_bytes = video_file.getvalue()
        st.session_state.step = "frame_selection"
        # Limpa resultados antigos se um novo vídeo for carregado
        st.session_state.results = None 
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
        if st.button("<< Frame Anterior"):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
    with col3:
        if st.button("Próximo Frame >>"):
            st.session_state.current_frame_idx = min(total_frames - 1, st.session_state.current_frame_idx + 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
    success, frame = cap.read()
    
    with col2:
        if success:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Exibindo Frame: {st.session_state.current_frame_idx} / {total_frames-1}")
        
    if st.button("Confirmar Frame e Iniciar Configuração", type="primary"):
        st.session_state.initial_frame = frame
        st.session_state.step = "configuration"
        st.rerun()
    
    cap.release()
    os.remove(video_path)

# --- PASSO 2: CONFIGURAÇÃO E ANÁLISE (INTEGRADO) ---
if st.session_state.step == "configuration":
    st.markdown("## Passo 3: Configuração e Análise")
    st.info("Use a grade para definir os parâmetros e clique em 'Iniciar Análise' para ver os resultados. Você pode ajustar os valores e reanalisar a qualquer momento.")

    frame_com_grade = desenhar_grade_cartesiana(st.session_state.initial_frame)
    altura_total, _, _ = frame_com_grade.shape
    
    col_config, col_preview = st.columns([1, 2])

    with col_config:
        st.markdown("#### 1. Definição da Origem (0,0)")
        orig_x = st.number_input("Origem - X", min_value=0, step=10, key="orig_x")
        orig_y_usuario = st.number_input("Origem - Y (contado de baixo)", min_value=0, step=10, key="orig_y")
        
        st.markdown("---")
        st.markdown("#### 2. Calibração da Escala")
        p1, p2 = st.columns(2)
        x1 = p1.number_input("Ponto 1 - X", min_value=0, step=10, key="x1")
        y1_usuario = p2.number_input("Ponto 1 - Y (de baixo)", min_value=0, step=10, key="y1")
        x2 = p1.number_input("Ponto 2 - X", min_value=0, step=10, key="x2")
        y2_usuario = p2.number_input("Ponto 2 - Y (de baixo)", min_value=0, step=10, key="y2")
        distancia_real = st.number_input("Distância real entre os pontos (em u.m.)", min_value=0.01, value=1.0, format="%.4f", key="dist_real")

        st.markdown("---")
        st.markdown("#### 3. Seleção do Objeto")
        obj_x = st.number_input("Objeto - X (canto esquerdo)", min_value=0, step=10, key="obj_x")
        obj_y_usuario = st.number_input("Objeto - Y (canto inferior)", min_value=0, step=10, key="obj_y")
        obj_w = st.number_input("Largura do Objeto", min_value=10, value=50, step=10, key="obj_w")
        obj_h = st.number_input("Altura do Objeto", min_value=10, value=50, step=10, key="obj_h")

        st.markdown("---")
        st.markdown("#### 4. Parâmetros de Geração")
        fator_dist = st.slider("Espaçamento na Imagem (u.m.)", 0.01, 2.0, 0.1, 0.01, help="Distância MÍNIMA (em u.m.) que o objeto precisa se mover para ser 'carimbado' na imagem final.")
        
        # Botão de processamento
        if st.button("🚀 Iniciar / Atualizar Análise", type="primary", use_container_width=True):
            status_text = st.empty()
            with st.spinner("Analisando o vídeo..."):
                # Conversões e Cálculos
                orig_y_opencv = altura_total - orig_y_usuario
                origin_coords = (orig_x, orig_y_opencv)
                
                y1_opencv = altura_total - y1_usuario
                y2_opencv = altura_total - y2_usuario
                length_pixels = np.sqrt((x2 - x1)**2 + (y2_opencv - y1_opencv)**2)
                
                if length_pixels > 0:
                    scale_factor = distancia_real / length_pixels
                    
                    obj_y_opencv = altura_total - obj_y_usuario - obj_h
                    bbox_opencv = (obj_x, obj_y_opencv, obj_w, obj_h)
                    
                    # Chama a função de processamento e armazena os resultados no estado da sessão
                    st.session_state.results = processar_video(
                        st.session_state.video_bytes,
                        st.session_state.initial_frame,
                        st.session_state.current_frame_idx,
                        bbox_opencv,
                        fator_dist,
                        scale_factor,
                        origin_coords,
                        status_text
                    )
                else:
                    st.error("A distância da escala em pixels não pode ser zero.")

    with col_preview:
        frame_para_preview = frame_com_grade.copy()
        
        # Desenha a Origem
        orig_y_opencv_preview = altura_total - orig_y_usuario
        cv2.circle(frame_para_preview, (orig_x, orig_y_opencv_preview), 10, (255, 0, 255), -1) # Magenta
        cv2.putText(frame_para_preview, "(0,0)", (orig_x + 15, orig_y_opencv_preview), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Desenha a Escala
        y1_opencv_preview = altura_total - y1_usuario
        y2_opencv_preview = altura_total - y2_usuario
        cv2.circle(frame_para_preview, (x1, y1_opencv_preview), 5, (0, 255, 255), -1) # Ciano
        cv2.circle(frame_para_preview, (x2, y2_opencv_preview), 5, (0, 255, 255), -1) # Ciano
        cv2.line(frame_para_preview, (x1, y1_opencv_preview), (x2, y2_opencv_preview), (0, 255, 255), 2)
        
        # Desenha o Bounding Box
        obj_y_opencv_preview = altura_total - obj_y_usuario - obj_h
        if obj_w > 0 and obj_h > 0:
            cv2.rectangle(frame_para_preview, (obj_x, obj_y_opencv_preview), (obj_x + obj_w, obj_y_opencv_preview + obj_h), (255, 0, 0), 2)
        
        st.image(cv2.cvtColor(frame_para_preview, cv2.COLOR_BGR2RGB), caption='Use a grade como referência para os parâmetros.')

    # --- Seção de Resultados (só aparece se houver resultados) ---
    if st.session_state.results:
        st.markdown("---")
        st.markdown("## ✅ Resultados da Análise")
        
        resultado_img, resultado_csv, figura_graficos = st.session_state.results

        if resultado_img is not None and resultado_csv is not None and figura_graficos is not None:
            with st.expander("Ver Resultados Detalhados", expanded=True):
                st.markdown("### Imagem Estroboscópica")
                st.image(resultado_img)
                st.download_button("💾 Baixar Imagem (.png)", resultado_img, "imagem_estroboscopica.png", "image/png", use_container_width=True)
                
                st.markdown("### Gráficos de Cinemática")
                st.pyplot(figura_graficos)
                
                st.markdown("### Tabela de Dados Completa")
                df_resultado = pd.read_csv(BytesIO(resultado_csv))
                st.dataframe(df_resultado)
                st.download_button("💾 Baixar Dados (CSV)", resultado_csv, "dados_trajetoria.csv", "text/csv", use_container_width=True)
        else:
            st.error("Falha na análise. O rastreador pode ter perdido o objeto. Verifique a seleção do objeto e tente novamente.")

    # Botão para recomeçar
    if st.button("Analisar outro vídeo"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
