# Desafio_Damas

Neste projeto, foi desenvolvido um sistema de detecção para jogo de damas utilizando visão computacional clássica com OpenCV-Python. O objetivo é identificar o status do tabuleiro e acompanhar as movimentações das peças através de um vídeo. O uso dos marcadores ArUcos foram recomendados para a calibração da perspectiva da câmera independente do posicionamento da câmera. Por fim, o status do jogo foram impressos no terminal via matriz numpy.

## Codificação com base na tarefa 'Mini-projeto visao.pdf'

### Classes e Funções


#### `ArucoDetector`
- **Objetivo**: Detectar marcadores ArUco no quadro do vídeo.
- **Métodos**:
  - `find_aruco_markers`: Localiza marcadores ArUco na imagem e, opcionalmente, desenha-os.
  - `find_closest_point_to_center`: Determina os pontos mais próximos ao centro de cada marcador.
  - `associate_points_with_ids`: Associa os pontos próximos aos IDs dos marcadores, baseado em um mapa predefinido.

#### `PerspectiveTransformer`
- **Objetivo**: Aplicar uma transformação de perspectiva ao quadro para alinhar o tabuleiro de forma que fique visível como um quadrado perfeito.
- **Métodos**:
  - `apply_transform`: Aplica a transformação de perspectiva utilizando os pontos associados aos marcadores ArUco.

#### `ObjectAndLineDetector`
- **Objetivo**: Detectar peças de damas coloridas no tabuleiro transformado.
- **Métodos**:
  - `detect_colored_objects`: Detecta objetos (peças) baseando-se em limiares de cor para verde e roxo.
  - `_draw_contours`: Desenha contornos ao redor dos objetos detectados.

#### Funções Auxiliares
- `draw_lines_and_labels`: Desenha linhas e rótulos das células no tabuleiro.
- `detect_board_status`: Avalia o estado atual do tabuleiro identificando a posição das peças.
- `calculate_average_board`: Calcula a média das detecções para estabilizar o reconhecimento ao longo do tempo.
- `process_frame`: Processa cada quadro do vídeo para detecção e análise.
- `board_frame`: Exibe o quadro processado na saída da 'janela'.
- `compare_boards`: Compara dois estados de tabuleiros para detectar reconhecimento válido.
- `detect_move`: Identifica movimentos das peças na mudança de estado do tabuleiro.
- `detect_winner`: Determina se há um vencedor com base no número de peças restantes.

### Programa Principal

- **Entrada de vídeo**: Aceita um arquivo de vídeo como entrada.
- **Resize e Crop**: Ajusta o tamanho e recorta o quadro para focar no tabuleiro.
- **Saídas**:
  - Exibe o vídeo original em uma janela.
  - Mostra o tabuleiro com o estado atual das peças em outra janela.
  - Imprime no terminal o estado do tabuleiro e informações sobre jogadas e peças capturadas, além de identificar o vencedor.

### Thiago Vilar