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
    carimbos_data = []
    posicao_ultimo_carimbo_px = (bbox_coords_opencv[0] + bbox_coords_opencv[2]/2, bbox_coords_opencv[1] + bbox_coords_opencv[3]/2)
    
    # Adiciona o ponto inicial aos carimbos para ter referência
    carimbos_data.append([
        start_frame_idx, 
        posicao_ultimo_carimbo_px[0], 
        posicao_ultimo_carimbo_px[1]
    ])

    contador_frames_processados = 0
    while True:
        frame_atual_idx = start_frame_idx + contador_frames_processados
        if frame_atual_idx >= total_frames: break

        success, frame_atual = cap.read()
        if not success: break
        
        status_text_element.text(f"Processando frame {frame_atual_idx}/{total_frames-1}...")
        
        success_track, bbox_atual = tracker.update(frame_atual)
        if success_track:
            centro_atual_px = (bbox_atual[0] + bbox_atual[2]/2, bbox_atual[1] + bbox_atual[3]/2)
            raw_data.append([frame_atual_idx, centro_atual_px[0], centro_atual_px[1]])
            
            dist_pixels = np.sqrt((centro_atual_px[0] - posicao_ultimo_carimbo_px[0])**2 + (centro_atual_px[1] - posicao_ultimo_carimbo_px[1])**2)
            
            if dist_pixels * scale_factor >= fator_distancia:
                (x, y, w, h) = [int(v) for v in bbox_atual]
                x_s, y_s, x_e, y_e = max(x, 0), max(y, 0), min(x + w, largura_frame), min(y + h, altura_frame)
                regiao = frame_atual[y_s:y_e, x_s:x_e]
                if regiao.size > 0:
                    imagem_estroboscopica[y_s:y_e, x_s:x_e] = regiao
                
                carimbos_data.append([frame_atual_idx, centro_atual_px[0], centro_atual_px[1]])
                posicao_ultimo_carimbo_px = centro_atual_px

        contador_frames_processados += 1
    
    cap.release()
    os.remove(video_path)
    
    if not raw_data: return None, None, None, None
    
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
    _, buffer_img_estrob = cv2.imencode('.PNG', imagem_estroboscopica)
    img_estrob_bytes = BytesIO(buffer_img_estrob).getvalue()
    
    figura_graficos = plotar_graficos(df_final)
    
    # Retorna também o dataframe e a lista de carimbos para a função de vetores
    return img_estrob_bytes, csv_bytes, figura_graficos, df_final, carimbos_data

def desenhar_vetores_velocidade(imagem_estroboscopica_original, df_completo, carimbos_data, origin_coords, scale_factor, scale_vetor, max_len_vetor):
    imagem_com_vetores = imagem_estroboscopica_original.copy()
    
    df_carimbos = pd.DataFrame(carimbos_data, columns=['frame', 'pos_x_px', 'pos_y_px'])
    
    # Usa o df completo para obter o tempo e as posições em u.m.
    df_merged = pd.merge(df_carimbos, df_completo, on='frame', how='left')

    df_merged['vx_um_s'] = df_merged['pos_x_um'].diff() / df_merged['tempo_s'].diff()
    df_merged['vy_um_s'] = df_merged['pos_y_um'].diff() / df_merged['tempo_s'].diff()

    for i in range(1, len(df_merged)):
        p_start_px = (int(df_merged.loc[i, 'pos_x_px']), int(df_merged.loc[i, 'pos_y_px']))
        
        vx = df_merged.loc[i, 'vx_um_s']
        vy = df_merged.loc[i, 'vy_um_s']
        vel_magnitude = np.sqrt(vx**2 + vy**2)

        if not np.isnan(vx) and not np.isnan(vy) and vel_magnitude > 0:
            arrow_length_pixels = min(max_len_vetor, vel_magnitude * scale_vetor) # scale_vetor agora é um multiplicador simples

            direction_x = vx / vel_magnitude
            direction_y = vy / vel_magnitude
            
            p_end_px = (int(p_start_px[0] + direction_x * arrow_length_pixels), 
                        int(p_start_px[1] - direction_y * arrow_length_pixels))
            
            cv2.arrowedLine(imagem_com_vetores, p_start_px, p_end_px, (0, 0, 255), 2, tipLength=0.3)
    
    _, buffer = cv2.imencode('.PNG', imagem_com_vetores)
    return BytesIO(buffer).getvalue()

# --- INTERFACE DA APLICAÇÃO ---

st.set_page_config(layout="wide", page_title="Análise de Movimento por Vídeo")
st.markdown("# 🔬 Análise de Movimento por Vídeo")
st.markdown("### Uma ferramenta para extrair dados cinemáticos de vídeos com câmera estática.")

# Inicializa o estado da sessão
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'results' not in st.session_state:
    st.session_state.results = None

# ... (O resto da interface continua o mesmo até a seção de resultados)
# --- PASSO 0: UPLOAD ---
if st.session_state.step == "upload":
    st.markdown("## Passo 1: Upload do Vídeo")
    video_file = st.file_uploader("Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    
    if video_file:
        st.session_state.video_bytes = video_file.getvalue()
        st.session_state.step = "frame_selection"
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
        
        if st.button("🚀 Iniciar / Atualizar Análise", type="primary", use_container_width=True):
            status_text = st.empty()
            with st.spinner("Analisando o vídeo..."):
                orig_y_opencv = altura_total - orig_y_usuario
                origin_coords = (orig_x, orig_y_opencv)
                
                y1_opencv = altura_total - y1_usuario
                y2_opencv = altura_total - y2_usuario
                length_pixels = np.sqrt((x2 - x1)**2 + (y2_opencv - y1_opencv)**2)
                
                if length_pixels > 0:
                    scale_factor = distancia_real / length_pixels
                    obj_y_opencv = altura_total - obj_y_usuario - obj_h
                    bbox_opencv = (obj_x, obj_y_opencv, obj_w, obj_h)
                    
                    st.session_state.results = processar_video(
                        st.session_state.video_bytes, st.session_state.initial_frame,
                        st.session_state.current_frame_idx, bbox_opencv, fator_dist,
                        scale_factor, origin_coords, status_text
                    )
                else:
                    st.error("A distância da escala em pixels não pode ser zero.")

    with col_preview:
        frame_para_preview = frame_com_grade.copy()
        orig_y_opencv_preview = altura_total - orig_y_usuario
        cv2.circle(frame_para_preview, (orig_x, orig_y_opencv_preview), 10, (255, 0, 255), -1)
        cv2.putText(frame_para_preview, "(0,0)", (orig_x + 15, orig_y_opencv_preview), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        y1_opencv_preview = altura_total - y1_usuario
        y2_opencv_preview = altura_total - y2_usuario
        cv2.circle(frame_para_preview, (x1, y1_opencv_preview), 5, (0, 255, 255), -1)
        cv2.circle(frame_para_preview, (x2, y2_opencv_preview), 5, (0, 255, 255), -1)
        cv2.line(frame_para_preview, (x1, y1_opencv_preview), (x2, y2_opencv_preview), (0, 255, 255), 2)
        
        obj_y_opencv_preview = altura_total - obj_y_usuario - obj_h
        if obj_w > 0 and obj_h > 0:
            cv2.rectangle(frame_para_preview, (obj_x, obj_y_opencv_preview), (obj_x + obj_w, obj_y_opencv_preview + obj_h), (255, 0, 0), 2)
        
        st.image(cv2.cvtColor(frame_para_preview, cv2.COLOR_BGR2RGB), caption='Use a grade como referência para os parâmetros.')

    if st.session_state.results:
        st.markdown("---")
        st.markdown("## ✅ Resultados da Análise Principal")
        
        img_estrob_bytes, resultado_csv, figura_graficos, df_final, carimbos_data = st.session_state.results

        if img_estrob_bytes is not None:
            with st.expander("Ver Resultados Detalhados", expanded=True):
                st.markdown("### Imagem Estroboscópica")
                st.image(img_estrob_bytes)
                st.download_button("💾 Baixar Imagem (.png)", img_estrob_bytes, "imagem_estroboscopica.png", "image/png", use_container_width=True)
                
                st.markdown("### Gráficos de Cinemática")
                st.pyplot(figura_graficos)
                
                st.markdown("### Tabela de Dados Completa")
                st.dataframe(df_final)
                st.download_button("💾 Baixar Dados (CSV)", resultado_csv, "dados_trajetoria.csv", "text/csv", use_container_width=True)

            # --- NOVA SEÇÃO DE ANÁLISE ADICIONAL ---
            st.markdown("---")
            with st.expander("Análise Adicional: Vetores de Velocidade"):
                st.info("Ajuste os parâmetros e clique no botão para gerar uma imagem com os vetores de velocidade.")
                
                vec_col1, vec_col2 = st.columns(2)
                scale_vetor = vec_col1.slider("Escala do Vetor", 1, 200, 50, help="Multiplicador para o comprimento dos vetores. Valores maiores = vetores mais longos.")
                max_len_vetor = vec_col2.slider("Comprimento Máximo (pixels)", 10, 200, 100, help="Limite máximo para o tamanho de um vetor na imagem.")
                
                if st.button("Gerar / Atualizar Imagem com Vetores", use_container_width=True):
                    # Chama a nova função de desenho, que é muito mais rápida
                    imagem_estroboscopica_original = cv2.imdecode(np.frombuffer(img_estrob_bytes, np.uint8), 1)
                    
                    img_vetores_bytes = desenhar_vetores_velocidade(
                        imagem_estroboscopica_original, df_final, carimbos_data,
                        st.session_state.origin_coords, st.session_state.scale_factor,
                        scale_vetor, max_len_vetor
                    )
                    st.session_state.img_vetores = img_vetores_bytes
                
                if 'img_vetores' in st.session_state and st.session_state.img_vetores is not None:
                    st.image(st.session_state.img_vetores, caption="Imagem Estroboscópica com Vetores de Velocidade")
                    st.download_button("💾 Baixar Imagem com Vetores (.png)", st.session_state.img_vetores, "imagem_com_vetores.png", "image/png", use_container_width=True)
        else:
            st.error("Falha na análise. O rastreador pode ter perdido o objeto.")

    if st.button("Analisar outro vídeo"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
