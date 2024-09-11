import cv2
import numpy as np
from cv2 import aruco

class ArucoDetector:
    def __init__(self, dictionary_type=aruco.DICT_ARUCO_ORIGINAL, id_map=None):
        self.dictionary_type = dictionary_type
        self.id_map = id_map or {}

    def find_aruco_markers(self, img, draw=True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(self.dictionary_type)
        aruco_params = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if draw and ids is not None:
            aruco.drawDetectedMarkers(img, corners, ids)
        return corners, ids

    def find_closest_point_to_center(self, img, corners):
        image_center = np.array([img.shape[1] // 2, img.shape[0] // 2])
        closest_points = []
        for corner in corners:
            distances = np.linalg.norm(corner[0] - image_center, axis=1)
            min_index = np.argmin(distances)
            closest_point = corner[0][min_index]
            closest_points.append(closest_point)
            cv2.circle(img, tuple(closest_point.astype(int)), 4, (0, 255, 0), -1)
        return np.array(closest_points)

    def associate_points_with_ids(self, closest_points, ids, img):
        required_ids = set(self.id_map.keys())
        if not required_ids.issubset(set(ids.flatten())):
            return None
        ordered_points = [None] * len(self.id_map)
        for i, point in enumerate(closest_points):
            aruco_id = ids[i][0]
            if aruco_id in self.id_map:
                item = self.id_map[aruco_id]
                position_index = item['position'][0] + item['position'][1] * 2
                ordered_points[position_index] = point
                cv2.putText(img, f"{item['label']} ({aruco_id})",
                            (int(point[0]), int(point[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if any(p is None for p in ordered_points):
            print("Erro: Falha ao associar corretamente os pontos aos IDs dos ArUcos.")
            return None
        return np.array(ordered_points)

class PerspectiveTransformer:
    def apply_transform(self, img, src_points):
        dst_points = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(src_points.astype('float32'), dst_points)
        warped = cv2.warpPerspective(img, matrix, (480, 480))
        return warped

class ObjectAndLineDetector:
    def __init__(self, green_thresholds, purple_thresholds, min_distance):
        self.green_thresholds = green_thresholds
        self.purple_thresholds = purple_thresholds
        self.min_distance = min_distance

    def detect_colored_objects(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        green_mask = cv2.inRange(hsv, np.array(self.green_thresholds[0], dtype="uint8"),
                                 np.array(self.green_thresholds[1], dtype="uint8"))
        purple_mask = cv2.inRange(hsv, np.array(self.purple_thresholds[0], dtype="uint8"),
                                  np.array(self.purple_thresholds[1], dtype="uint8"))
        green_centers = self._draw_contours(img, green_mask, (0, 255, 0))
        purple_centers = self._draw_contours(img, purple_mask, (128, 0, 128))
        # Filtra centros para evitar sobreposição entre verdes e roxos
        filtered_green_centers = [g for g in green_centers if all(np.linalg.norm(g - p) > self.min_distance for p in purple_centers)]
        filtered_purple_centers = [p for p in purple_centers if all(np.linalg.norm(p - g) > self.min_distance for g in green_centers)]
        return filtered_green_centers, filtered_purple_centers

    def _draw_contours(self, img, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                point = np.array([cx, cy])
                if all(np.linalg.norm(point - np.array(center)) > self.min_distance for center in centers):
                    centers.append(point)
                    cv2.circle(img, (cx, cy), 15, color, 2)
        return centers

def draw_lines_and_labels(warped):
    text_color = (0, 0, 0)
    cell_size = 60
    for i in range (8):
        for j in range (8):
            if (i + j) % 2 == 1:
                cell_label = chr(65 + j) + str(8 - i)
                x_pos = j * cell_size + cell_size // 2 - 10
                y_pos = i * cell_size + cell_size // 2 + 5
                cv2.putText(warped, cell_label, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return warped

def detect_board_status(warped, green_centers, purple_centers):
    board = np.full((8, 8), '⬛', dtype=object)
    cell_size = 60
    emoji_map = {1: '🟢', 2: '🟣'}
    for i in range (8):
        for j in range (8):
            if (i + j) % 2 == 1:
                cell_x = j * cell_size + cell_size // 2
                cell_y = i * cell_size + cell_size // 2
                for center in green_centers:
                    if abs(center[0] - cell_x) < cell_size // 2 and abs(center[1] - cell_y) < cell_size // 2:
                        board[i, j] = emoji_map[1]
                for center in purple_centers:
                    if abs(center[0] - cell_x) < cell_size // 2 and abs(center[1] - cell_y) < cell_size // 2:
                        board[i, j] = emoji_map[2]
            else:
                board[i, j] = '⬜'
    return board

def calculate_average_board(board_accumulator):
    final_board = np.full((8, 8), '⬛', dtype=object)
    for i in range (8):
        for j in range (8):
            cell_votes = [board[i][j] for board in board_accumulator if board[i][j] in ['🟢', '🟣']]
            if cell_votes:
                final_board[i, j] = max(set(cell_votes), key=cell_votes.count)
            else:
                final_board[i, j] = '⬜' if (i + j) % 2 == 1 else '⬛'
    return final_board

def process_frame(frame, detector, transformer, object_line_detector, previous_board, board_accumulator):
    corners, ids = detector.find_aruco_markers(frame)
    if ids is not None and corners:
        closest_points = detector.find_closest_point_to_center(frame, corners)
        labeled_points = detector.associate_points_with_ids(closest_points, ids, frame)
        if labeled_points is not None:
            warped = transformer.apply_transform(frame, labeled_points)
            green_centers, purple_centers = object_line_detector.detect_colored_objects(warped)
            current_board = detect_board_status(warped, green_centers, purple_centers)
            board_accumulator.append(current_board)

            if len(board_accumulator) >= 50:
                average_board = calculate_average_board(board_accumulator)
                board_accumulator.clear()

                detect_winner(average_board)

                if previous_board is not None:
                    move_made = detect_move(previous_board, average_board)
                    if move_made:
                        print("Tabuleiro atualizado:")
                        print(average_board)
                        previous_board = average_board
                else:
                    print("Tabuleiro inicial:")
                    print(average_board)
                    previous_board = average_board

    return frame, previous_board

def board_frame(frame, detector, transformer, object_line_detector, previous_board):
    corners, ids = detector.find_aruco_markers(frame)
    if ids is not None and corners:
        closest_points = detector.find_closest_point_to_center(frame, corners)
        labeled_points = detector.associate_points_with_ids(closest_points, ids, frame)
        if labeled_points is not None:
            warped = transformer.apply_transform(frame, labeled_points)
            green_centers, purple_centers = object_line_detector.detect_colored_objects(warped)
            current_board = detect_board_status(warped, green_centers, purple_centers)
            final_image = draw_lines_and_labels(warped)
            cv2.imshow('Processed Image', final_image)
            cv2.waitKey(1)
            if previous_board is not None:
                detect_move(previous_board, current_board)
            previous_board = current_board
            return frame, previous_board
    return frame, previous_board


def compare_boards(board1, board2):
    differences = []
    for i in range(8):
        for j in range(8):
            if board1[i, j] != board2[i, j]:
                differences.append((i, j, board1[i, j], board2[i, j]))
    return differences

def detect_move(previous_board, current_board):
    differences = compare_boards(previous_board, current_board)
    moves = []
    captured_pieces = []

    # Detectar movimentos e capturas
    for diff in differences:
        i, j, prev, curr = diff
        if prev in ['🟢', '🟣'] and curr == '⬜':  # Possível movimento ou captura
            moved = False
            for diff_next in differences:
                i_next, j_next, prev_next, curr_next = diff_next
                if curr_next in ['🟢', '🟣'] and prev_next == '⬜' and curr_next == prev:  # Confirma movimento
                    moves.append((curr_next, (i, j), (i_next, j_next)))
                    moved = True
                    break
            if not moved:  # Se confirmado movimento = capturada
                captured_pieces.append((prev, (i, j)))

    # Remove duplicatas nos movimentos
    unique_moves = []
    seen_moves = set()
    for move in moves:
        if move not in seen_moves:
            seen_moves.add(move)
            unique_moves.append(move)

    # Anunciar movimentos
    for move in unique_moves:
        piece, origin, destination = move
        origin_label = chr(65 + origin[1]) + str(8 - origin[0])
        destination_label = chr(65 + destination[1]) + str(8 - destination[0])
        print(f"Jogador {piece} moveu peça de {origin_label} para {destination_label}")

    # Anunciar capturas
    for piece, location in captured_pieces:
        location_label = chr(65 + location[1]) + str(8 - location[0])
        print(f"Peça {piece} capturada em {location_label}")

    return unique_moves if unique_moves else None

def detect_winner(board):
    green_count = sum(piece == '🟢' for row in board for piece in row)
    purple_count = sum(piece == '🟣' for row in board for piece in row)
    if green_count == 0:
        print("Jogador PURPLE venceu!")
    elif purple_count == 0:
        print("Jogador GREEN venceu!")
    # else:
        # print(f"Estado do jogo - Verde: {green_count}, Roxo: {purple_count}")

def main():
    aruco_id_map = {
        2: {'label': 'P1', 'position': (0, 0)},
        10: {'label': 'P2', 'position': (1, 0)},
        11: {'label': 'P3', 'position': (1, 1)},
        12: {'label': 'P4', 'position': (0, 1)},
    }

    green_thresholds = ([88, 180, 104], [135, 255, 187])
    purple_thresholds = ([118, 100, 66], [255, 251, 255])
    min_distance = 55


    detector = ArucoDetector(id_map=aruco_id_map)
    transformer = PerspectiveTransformer()
    object_line_detector = ObjectAndLineDetector(green_thresholds, purple_thresholds, min_distance)
    
    cap = cv2.VideoCapture("2024-09-03 14-31-00.mp4")

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
     
    previous_board = None  
    board_accumulator = [] 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Vídeo finalizado ou erro ao ler o frame.")
            break
        
        #SAÍDA DO VÍDEO CROPPADO   
        frame_resized = cv2.resize(frame, (0, 0), fx=0.67, fy=0.67)
        height, width = frame_resized.shape[:2]
        video_cropped = frame_resized[:, 110:width - 110]  
        cv2.imshow("video", video_cropped)
        
        #PROCESSAMENTO COM IMPRESSÃO NUMPY NO TERMINAL
        frame_resized, previous_board = process_frame(frame_resized, detector, transformer, object_line_detector, previous_board, board_accumulator)

        #SAÍDA DE TABULEIRO COM DETECÇÕES - TELA
        board_frame(frame, detector, transformer, object_line_detector, previous_board)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
