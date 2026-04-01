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
    """Gera gráficos de velocidade e aceleração usando Matplotlib."""
    # Alterado de (2, 1) para (3, 1) e ajustado o figsize para acomodar o terceiro gráfico
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    plt.subplots_adjust(hspace=0.4) # Aumentado o espaçamento vertical entre os gráficos
    
    # Gráfico 1: Velocidade X
    ax1.plot(df['tempo_s'], df['vx_um_s'], marker='o', linestyle='-', markersize=4)
    ax1.set_title("Velocidade Horizontal (Vx)")
    ax1.set_ylabel("Velocidade (u.m./s)")
    ax1.grid(True)
    
    # Gráfico 2: Velocidade Y
    ax2.plot(df['tempo_s'], df['vy_um_s'], marker='o', linestyle='-', markersize=4)
    ax2.set_title("Velocidade Vertical (Vy)")
    ax2.set_ylabel("Velocidade (u.m./s)")
    ax2.grid(True)
    
    # --- NOVO BLOCO: Gráfico 3: Aceleração Total ---
    ax3.plot(df['tempo_s'], df['acel_total_um_s2'], marker='^', linestyle='-', markersize=4) # Marcador diferente
    ax3.set_title("Magnitude da Aceleração Total")
    ax3.set_ylabel("Aceleração (u.m./s²)")
    ax3.set_xlabel("Tempo (s)") # O eixo X só é rotulado no gráfico inferior
    ax3.grid(True)
    # -----------------------------------------------
    
    # Adicionar uma linha de referência para g (~9.8) se a aceleração estiver próxima
    # (Opcional, mas muito didático para queda livre)
    promedio_acel = df['acel_total_um_s2'].mean()
    if 8 < promedio_acel < 11:
        ax3.axhline(y=promedio_acel, linestyle='--', color='gray', alpha=0.5, label='Acel. Média')
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

def processar_video(video_bytes, initial_frame, start_frame_idx, bbox_coords_opencv, fator_distancia, scale_factor, origin_coords, status_text_element, window_size=11, poly_order=2):
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
    
    carimbos_data = []
    posicao_ultimo_carimbo_px = (bbox_coords_opencv[0] + bbox_coords_opencv[2]/2, bbox_coords_opencv[1] + bbox_coords_opencv[3]/2)
    carimbos_data.append([start_frame_idx, posicao_ultimo_carimbo_px[0], posicao_ultimo_carimbo_px[1]])

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
    
    cap.release(), os.remove(video_path)
    if len(carimbos_data) < 2: return None
    
    df_carimbos = pd.DataFrame(carimbos_data, columns=['frame', 'pos_x_px', 'pos_y_px'])
    df_carimbos['tempo_s'] = (df_carimbos['frame'] - start_frame_idx) / fps
    df_carimbos['pos_x_um'] = (df_carimbos['pos_x_px'] - origin_coords[0]) * scale_factor
    df_carimbos['pos_y_um'] = -(df_carimbos['pos_y_px'] - origin_coords[1]) * scale_factor

    # Cálculo de Velocidade (Savitzky-Golay vs Diferenças Finitas) 
    if len(df_carimbos) > window_size:
        dt = 1.0 / fps
        
        # 1. Posição Suavizada (deriv=0)
        df_carimbos['pos_x_um'] = savgol_filter(df_carimbos['pos_x_um'], window_length=window_size, polyorder=poly_order, deriv=0)
        df_carimbos['pos_y_um'] = savgol_filter(df_carimbos['pos_y_um'], window_length=window_size, polyorder=poly_order, deriv=0)
        
        # 2. Velocidade (deriv=1)
        df_carimbos['vx_um_s'] = savgol_filter(df_carimbos['pos_x_um'], window_length=window_size, polyorder=poly_order, deriv=1, delta=dt)
        df_carimbos['vy_um_s'] = savgol_filter(df_carimbos['pos_y_um'], window_length=window_size, polyorder=poly_order, deriv=1, delta=dt)
        
        # 3. Componentes da Aceleração (deriv=2)
        ax = savgol_filter(df_carimbos['pos_x_um'], window_length=window_size, polyorder=poly_order, deriv=2, delta=dt)
        ay = savgol_filter(df_carimbos['pos_y_um'], window_length=window_size, polyorder=poly_order, deriv=2, delta=dt)
        
        # 4. Magnitude Total da Aceleração (Resultante)
        df_carimbos['acel_total_um_s2'] = np.sqrt(ax**2 + ay**2) # <--- ADICIONE ESTA LINHA
        
    else:
        delta_t = df_carimbos['tempo_s'].diff()
        df_carimbos['vx_um_s'] = df_carimbos['pos_x_um'].diff() / delta_t
        df_carimbos['vy_um_s'] = df_carimbos['pos_y_um'].diff() / delta_t
        
        # Fallback simples para aceleração por diferenças finitas
        df_carimbos['acel_total_um_s2'] = df_carimbos['vy_um_s'].diff() / delta_t # <--- ADICIONE ESTA LINHA
    
    df_final = df_carimbos.fillna(0)
    
    status_text_element.success(f"Processamento concluído! {len(df_final)} pontos de dados analisados.")
    
    _, buffer_img_estrob = cv2.imencode('.PNG', imagem_estroboscopica)
    img_estrob_bytes = BytesIO(buffer_img_estrob).getvalue()
    
    figura_graficos = plotar_graficos(df_final)
    
    return img_estrob_bytes, df_final, figura_graficos

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

    video_file = st.file_uploader("Escolha um arquivo de vídeo (MP4, AVI, MOV)", type=["mp4", "avi", "mov"], label_visibility="collapsed")
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
            st.number_input(
                "Ir para o Frame:",
                min_value=0,
                max_value=total_frames - 1,
                step=1,
                key="current_frame_idx",
                label_visibility="visible"
            )

    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
    success, frame = cap.read()
    if success: 
        img_cols = st.columns([1, 8, 1])
        with img_cols[1]:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Exibindo Frame: {st.session_state.current_frame_idx} / {total_frames-1}")
        
    if st.button("Confirmar Frame e Iniciar Configuração", type="primary"):
        st.session_state.initial_frame = frame
        # --- CORREÇÃO AQUI: Salva o frame de início permanentemente ---
        st.session_state.start_frame_for_analysis = st.session_state.current_frame_idx
        st.session_state.step = "configuration"
        st.rerun()
    
    cap.release(), os.remove(tfile.name)

if st.session_state.step == "configuration":
    st.markdown("## Passo 3: Configuração e Análise")
    st.info("Use a grade para definir os parâmetros e clique em 'Iniciar Análise'. Você pode ajustar os valores e reanalisar a qualquer momento.")
    frame_com_grade = desenhar_grade_cartesiana(st.session_state.initial_frame)
    altura_total, _, _ = frame_com_grade.shape
    col_config, col_preview = st.columns([1, 2])

    with col_config:
        st.markdown("#### 1. Definição da Origem (0,0)")
        orig_x = st.number_input("Origem - X", 0, step=10, key="orig_x")
        orig_y_usuario = st.number_input("Origem - Y", 0, step=10, key="orig_y")
        st.markdown("---")
        st.markdown("#### 2. Calibração da Escala")
        p1, p2 = st.columns(2)
        x1 = p1.number_input("Ponto 1 - X", 0, step=10, key="x1")
        y1_usuario = p2.number_input("Ponto 1 - Y", 0, step=10, key="y1")
        x2 = p1.number_input("Ponto 2 - X", 0, step=10, key="x2")
        y2_usuario = p2.number_input("Ponto 2 - Y", 0, step=10, key="y2")
        distancia_real = st.number_input("Distância real entre os pontos (em u.m.)", min_value=0.01, format="%.4f", key="dist_real")
        st.markdown("---")
        st.markdown("#### 3. Seleção do Objeto")
        obj_x = st.number_input("Objeto - X (canto esquerdo)", 0, step=10, key="obj_x")
        obj_y_usuario = st.number_input("Objeto - Y (canto inferior)", 0, step=10, key="obj_y")
        obj_w = st.number_input("Largura do Objeto", min_value=10, value=50, step=10, key="obj_w")
        obj_h = st.number_input("Altura do Objeto", min_value=10, value=50, step=10, key="obj_h")
        st.markdown("---")
        st.markdown("#### 4. Parâmetros de Geração")
        fator_dist = st.slider("Espaçamento na Imagem (u.m.)", 0.01, 5.0, 0.5, 0.01)

        st.markdown("---")
        st.markdown("#### 5. Suavização Cinemática (Savitzky-Golay)")
        window_size = st.slider("Tamanho da Janela", min_value=5, max_value=51, value=11, step=2, help="Deve ser ímpar. Valores maiores geram curvas mais suaves, mas podem atenuar picos reais de velocidade.")
        poly_order = st.slider("Ordem do Polinômio", min_value=1, max_value=4, value=2, help="Grau do polinômio ajustado. 2 (parábola) é ideal para movimentos com aceleração constante.")
        
        if window_size <= poly_order:
            st.error("Erro: O tamanho da janela deve ser maior que a ordem do polinômio.")
        
        if st.button("🚀 Iniciar / Atualizar Análise", type="primary", use_container_width=True):
            status_text = st.empty()
            with st.spinner("Analisando o vídeo..."):
                orig_y_opencv, y1_opencv, y2_opencv = altura_total - orig_y_usuario, altura_total - y1_usuario, altura_total - y2_usuario
                origin_coords = (orig_x, orig_y_opencv)
                length_pixels = np.sqrt((x2 - x1)**2 + (y2_opencv - y1_opencv)**2)
                if length_pixels > 0:
                    scale_factor = distancia_real / length_pixels
                    obj_y_opencv = altura_total - obj_y_usuario - obj_h
                    bbox_opencv = (obj_x, obj_y_opencv, obj_w, obj_h)
                    st.session_state.origin_coords, st.session_state.scale_factor = origin_coords, scale_factor
                    
                    header_comentarios = (
                        f"# Análise de Movimento - {pd.Timestamp.now()}\n"
                        f"# Parâmetros de Entrada:\n"
                        # --- CORREÇÃO AQUI: Usa a variável permanente ---
                        f"# Frame Inicial: {st.session_state.start_frame_for_analysis}\n"
                        f"# Origem (pixels, de cima): {origin_coords}\n"
                        f"# Fator de Escala: {scale_factor:.6f} u.m./pixel\n"
                        f"# Objeto Rastreado (x, y, w, h): {bbox_opencv}\n"
                        f"# --- \n"
                    )

                    st.session_state.results = processar_video(
                        st.session_state.video_bytes, 
                        st.session_state.initial_frame, 
                        st.session_state.start_frame_for_analysis, 
                        bbox_opencv, fator_dist, scale_factor, origin_coords, status_text,
                        window_size, poly_order
                    )
                    
                    st.session_state.csv_header = header_comentarios
                else: st.error("A distância da escala em pixels não pode ser zero.")

    with col_preview:
        frame_para_preview = frame_com_grade.copy()
        orig_y_opencv_preview, y1_opencv_preview, y2_opencv_preview = altura_total - orig_y_usuario, altura_total - y1_usuario, altura_total - y2_usuario
        cv2.circle(frame_para_preview, (orig_x, orig_y_opencv_preview), 10, (255, 0, 255), -1)
        cv2.putText(frame_para_preview, "(0,0)", (orig_x + 15, orig_y_opencv_preview), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.circle(frame_para_preview, (x1, y1_opencv_preview), 5, (0, 255, 255), -1)
        cv2.circle(frame_para_preview, (x2, y2_opencv_preview), 5, (0, 255, 255), -1)
        cv2.line(frame_para_preview, (x1, y1_opencv_preview), (x2, y2_opencv_preview), (0, 255, 255), 2)
        obj_y_opencv_preview = altura_total - obj_y_usuario - obj_h
        if obj_w > 0 and obj_h > 0: cv2.rectangle(frame_para_preview, (obj_x, obj_y_opencv_preview), (obj_x + obj_w, obj_y_opencv_preview + obj_h), (255, 0, 0), 2)
        
        st.image(cv2.cvtColor(frame_para_preview, cv2.COLOR_BGR2RGB), caption='Use a grade como referência para os parâmetros.')
        
        #_, buffer_preview = cv2.imencode('.PNG', cv2.cvtColor(frame_para_preview, cv2.COLOR_RGB2BGR))
        _, buffer_preview = cv2.imencode('.PNG', frame_para_preview)
        preview_bytes = BytesIO(buffer_preview).getvalue()
        st.download_button("💾 Baixar Imagem de Configuração", preview_bytes, "imagem_configuracao.png", "image/png", use_container_width=True)

    if st.session_state.results:
        st.markdown("---")
        st.markdown("## ✅ Resultados da Análise Principal")
        img_estrob_bytes, df_final, figura_graficos = st.session_state.results

        if img_estrob_bytes:
            with st.expander("Resultados Detalhados", expanded=True):
                st.markdown("### Imagem Estroboscópica"); st.image(img_estrob_bytes)
                st.download_button("💾 Baixar Imagem (.png)", img_estrob_bytes, "imagem_estroboscopica.png", "image/png", use_container_width=True)
                st.markdown("### Gráficos de Cinemática"); st.pyplot(figura_graficos)
                st.markdown("### Tabela de Dados"); st.dataframe(df_final)
                
                csv_final_string = st.session_state.csv_header + df_final.to_csv(index=False)
                st.download_button("💾 Baixar Dados (CSV)", csv_final_string, "dados_analise.csv", "text/csv", use_container_width=True)

            st.markdown("---")
            with st.expander("Análise Adicional: Vetores de Velocidade"):
                st.info("Ajuste os parâmetros e clique para gerar uma imagem com os vetores de velocidade.")
                cores_bgr = {"Vermelho": (0, 0, 255), "Azul": (255, 0, 0), "Amarelo": (0, 255, 255), "Ciano": (255, 255, 0), "Magenta": (255, 0, 255), "Verde": (0, 255, 0), "Branco": (255, 255, 255), "Laranja": (0, 165, 255)}
                
                vec_col1, vec_col2 = st.columns(2)
                cor_nome = vec_col1.selectbox("Cor do Vetor", options=list(cores_bgr.keys()))
                espessura_vetor = vec_col2.slider("Espessura do Vetor (px)", 1, 5, 2)
                scale_vetor_col, max_len_col = st.columns(2)
                scale_vetor = scale_vetor_col.slider("Escala do Vetor", 1, 200, 50, help="Multiplicador para o comprimento dos vetores.")
                max_len_vetor = max_len_col.slider("Comprimento Máximo (px)", 10, 200, 100, help="Limite para o tamanho de um vetor na imagem.")
                
                if st.button("Gerar / Atualizar Imagem com Vetores", use_container_width=True):
                    imagem_original = cv2.imdecode(np.frombuffer(img_estrob_bytes, np.uint8), 1)
                    img_vetores_bytes = desenhar_vetores_velocidade(imagem_original, df_final, scale_vetor, max_len_vetor, cores_bgr[cor_nome], espessura_vetor)
                    st.session_state.img_vetores = img_vetores_bytes
                
                if 'img_vetores' in st.session_state and st.session_state.img_vetores:
                    st.image(st.session_state.img_vetores, caption="Imagem Estroboscópica com Vetores de Velocidade")
                    st.download_button("💾 Baixar Imagem com Vetores (.png)", st.session_state.img_vetores, "imagem_com_vetores.png", "image/png", use_container_width=True)
            
            st.markdown("---")
            if st.button("🔄 Analisar outro vídeo", key="final_reset"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.error("Falha na análise. Nenhum ponto de dados foi gerado. Verifique se o objeto se move o suficiente para o 'Espaçamento na Imagem'.")

# --- Rodapé Informativo ---
st.markdown("---")
footer_expander = st.expander("💡 Informações Adicionais (Funcionalidades e Dicas)", expanded=False)
with footer_expander:
    st.markdown("#### Funcionalidades")
    st.markdown("""
    - **Seleção de Frame Inicial:** Navegue pelo vídeo para definir o ponto exato de início da análise.
    - **Calibração de Espaço:** Defina uma origem (0,0) e uma escala de referência (u.m./pixel) para obter dados em unidades de medida reais.
    - **Rastreamento de Objeto:** Acompanha o objeto selecionado ao longo do vídeo.
    - **Análise Cinemática:** Calcula e exibe dados de posição e velocidade (componentes X/Y).
    - **Visualização de Dados:** Gera uma imagem estroboscópica, gráficos de trajetória/velocidade e uma imagem opcional com vetores de velocidade.
    - **Exportação de Resultados:** Permite o download da imagem estroboscópica e da tabela de dados completa em formato CSV, incluindo os parâmetros da análise.
    """)
    st.markdown("#### Dicas para Melhores Resultados")
    st.markdown("""
    - **Câmera Estritamente Estática:** Para um resultado preciso, é fundamental que o vídeo tenha sido gravado com a **câmera completamente parada**. Qualquer movimento pode comprometer a análise.
    - **Dados Precisos (Visão Lateral):** Para melhor acurácia dos dados, a gravação deve ser realizada em um **plano paralelo ao do movimento do objeto** (visão 2D, de lado). Evite gravar o objeto se movendo em direção ou para longe da câmera.
    - **Bom Contraste:** Vídeos onde o objeto se destaca do fundo (cores diferentes, boa iluminação) produzem um rastreamento mais confiável.
    - **Seleção Precisa:** Dedique um momento para ajustar o retângulo de seleção azul para que ele envolva o objeto de forma justa na sua posição inicial. Uma seleção precisa é a chave para um rastreamento bem-sucedido.
    """)
