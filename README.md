# 🔬 Gerador de Imagem Estroboscópica e Dados de Trajetória

Este projeto foi desenvolvido como parte de um trabalho de conclusão de curso do Mestrado Profissional em Matemática em Rede Nacional (PROFMAT) na Universidade Federal de Uberlândia (UFU).

---

## Autoria e Orientação

* **Aluno de Mestrado:** Antônio Marcos da Silva Leite
* **Professor Orientador:** Prof. Dr. Rafael Figueiredo
* **Instituição:** Instituto de Matemática e Estatística da Universidade Federal de Uberlândia (IME-UFU)
* **Programa:** Mestrado Profissional em Matemática em Rede Nacional (PROFMAT)

---

Uma aplicação web construída com Streamlit e OpenCV para analisar o movimento de objetos em vídeos. A ferramenta gera uma imagem estroboscópica que visualiza a trajetória do objeto e exporta dados de posição frame a frame para um arquivo CSV.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)

---

## 🚀 Link para a Aplicação Web

**[Acesse a aplicação aqui!](https://stroboscopic-pvy4jugzxv6nnfkzmogyhg.streamlit.app/)**

---

## ✨ Funcionalidades

- **Upload de Vídeo:** Suporte para os formatos de vídeo mais comuns (MP4, AVI, MOV).
- **Seleção Interativa de Objeto:** Uma grade de referência cartesiana (origem no canto inferior esquerdo) e uma pré-visualização em tempo real do *bounding box* permitem uma seleção precisa do objeto de interesse.
- **Geração de Imagem Estroboscópica:** Cria uma única imagem composta que mostra o objeto em múltiplas posições ao longo do tempo.
- **Exportação de Dados de Trajetória:** Gera um arquivo `.csv` com a posição (relativa e absoluta) do centro do objeto em cada frame do vídeo.
- **Parâmetros Ajustáveis:** Controle o espaçamento entre as "impressões" do objeto na imagem final para diferentes efeitos visuais.

---

## 💡 Dicas para Melhores Resultados

-   **Câmera Estritamente Estática:** Para um resultado preciso, é fundamental que o vídeo tenha sido gravado com a **câmera completamente parada**. Qualquer movimento, vibração ou ajuste de zoom na câmera durante a gravação pode interferir na lógica de rastreamento e comprometer a qualidade da imagem e dos dados gerados.
-   **Bom Contraste:** Vídeos onde o objeto em movimento tem um bom contraste em relação ao fundo tendem a produzir resultados mais confiáveis.
-   **Seleção Precisa:** Dedique um momento para ajustar o retângulo de seleção azul para que ele envolva o objeto de forma justa na sua posição inicial. Uma seleção precisa é a chave para um rastreamento bem-sucedido.

---

## 🛠️ Tecnologias Utilizadas

- **Backend:** Python
- **Processamento de Imagem e Vídeo:** OpenCV (`opencv-contrib-python`)
- **Interface Web:** Streamlit
- **Manipulação de Dados:** Pandas & NumPy
- **Hospedagem:** Streamlit Community Cloud

---

## 📂 Estrutura do Projeto
├── stroboscopic_app.py   # O código principal da aplicação Streamlit
├── requirements.txt      # Dependências Python (pip)
├── packages.txt          # Dependências de sistema (apt-get para o servidor)
├── LICENSE               # Licença de uso
└── README.md             # Este arquivo

---

## 🖥️ Como Executar Localmente

Para executar esta aplicação em sua máquina local, siga os passos abaixo.

### Pré-requisitos

- Git
- Python 3.9 ou superior
- `pip` e `venv`

### Passos

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Rafael-UFU/Stroboscopic.git](https://github.com/Rafael-UFU/Stroboscopic.git)
    cd Stroboscopic
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Para macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências de sistema (apenas para Linux):**
    A aplicação depende de algumas bibliotecas que precisam ser instaladas. Em sistemas baseados em Debian/Ubuntu:
    ```bash
    sudo apt-get update
    sudo apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx
    ```

4.  **Instale as dependências Python:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Execute a aplicação Streamlit:**
    ```bash
    streamlit run stroboscopic_app.py
    ```
    A aplicação será aberta automaticamente no seu navegador.

## 📄 Licença

Este projeto está licenciado sob a **Licença MIT**. Veja o arquivo `LICENSE` no repositório para mais detalhes.

## 👨‍💻 Autor

- **[Rafael-UFU](https://github.com/Rafael-UFU)**


