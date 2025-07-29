import cv2
from ultralytics import YOLO

def map_to_board_index(center_x, center_y, top_left, square_size):
    x0, y0 = top_left
    col = int((center_x - x0) // square_size)
    rw = int((center_y - y0) // square_size)
    col = min(max(col, 0), 7)
    rw = min(max(rw, 0), 7)
    return [rw, col]

def board_to_fen(board):
    fen_rows = []
    for r in board:
        fen_row = ""
        empty = 0
        for cell in r:
            if cell == "":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def predict_fen_code(img_url):
    model = YOLO("best.pt")

    class_names = ['b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']
    img = cv2.imread(img_url)
    results = model(img)
    boxes = results[0].boxes.xyxy
    confidences = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    centers = []
    position = []
    box_width = 0
    top_x = 100000
    left_y = 100000
    for i, box in enumerate(boxes):
        confidence = confidences[i]
        class_id = int(class_ids[i])

        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 0)

        if class_names[class_id] == "Q" or class_names[class_id] == "q":
            x_err = (x2 - x1)
            y_err = (y2 - y1)
            cv2.rectangle(img, (x1 - int(x_err * 0.05), y1 - int(y_err * 0.05)),
                          (x2 + int(x_err * 0.05), y2 + int(y_err * 0.05)), (0, 255, 0), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if int(confidence * 100) > 70:
            top_x = min((x1 + x2) / 2, top_x)
            left_y = min((y1 + y2) / 2, left_y)
            centers.append([((x1 + x2) / 2, (y1 + y2) / 2), class_names[class_id]])
            if class_names[class_id] == "Q" or class_names[class_id] == "q":
                box_width = (x2 - x1) + int((x2 - x1) * 0.1)
            label = f"{class_names[class_id]} {int(confidence * 100)}"
            cv2.putText(img, label, (x1, int((y1 - 10 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 1)
            

    # Display the image with bounding boxes
    cv2.circle(img, (int(top_x - (box_width / 2)), int(left_y - (box_width / 2))), radius=5, color=(0, 0, 255),thickness=-1)

    top_x = top_x - (box_width / 2)
    left_y = left_y - (box_width / 2)
    perm_x = top_x
    perm_y = left_y

    for i in range(8):
        temp = []
        for j in range(8):
            cv2.rectangle(img, (int(top_x), int(left_y)), (int(top_x + box_width), int(left_y + box_width)),(0, 0, 255), 3)
            top_x = top_x + box_width
            temp.append("")
        top_x = perm_x
        left_y = left_y + box_width
        position.append(temp)

    for cord, name in centers:
        idx = map_to_board_index(cord[0], cord[1], [perm_x, perm_y], box_width)
        position[idx[0]][idx[1]] = name
        piece_map = {
        "b": "BB", "k": "BK", "n": "BN", "p": "BP", "q": "BQ", "r": "BR",
        "B": "WB", "K": "WK", "N": "WN", "P": "WP", "Q": "WQ", "R": "WR"
        }

        text = piece_map[name]
        text_x = int(cord[0])
        text_y = int(cord[1])
        cv2.putText(img, text, (text_x - 20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        
    side = int(input("what is your side 1 for white and 0 for black ?"))

    if side == 0:
        idx = 0
        for pos in position:
            position[idx] = pos[::-1]
            idx += 1
        board_for_fen = position[::-1]

        position_fen = board_to_fen(board_for_fen)

        for row in board_for_fen:
            print(row)
        final_fen = f"{position_fen} b - - 0 1"
        print("FEN:", final_fen)
        cv2.imwrite("labeled_board.jpg", img)
        # saving the annoted image
        return  final_fen
    else:
        position_fen = board_to_fen(position)
        final_fen = f"{position_fen} w - - 0 1"
        # saving the annoted image
        cv2.imwrite("labeled_board.jpg", img)
        print("FEN:", final_fen)
        return final_fen


predict_fen_code("C:\\Users\\merak\\Downloads\\drive-download-20250729T062149Z-1-001\\images\\val\\998a3417-20250410_232229.png")