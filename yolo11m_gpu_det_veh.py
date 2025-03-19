import cv2
import itertools as itools
from ultralytics import YOLO


# Constants
RECT_X1, RECT_Y1, RECT_X2, RECT_Y2 = 650, 150, 950, 700
COUNTER_TEXT_OFFSET = 34
FONT = cv2.FONT_HERSHEY_SIMPLEX

def load_model(model_path, device = 'cuda'):
    """
    Load YOLO model to the specified device.

    Args:
        model_path (str): Path to the YOLO model file.
        device (str, optional): Device to load the model to ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        YOLO: Loaded YOLO model.
    """
    try:
        model = YOLO(model_path).to(device)

        return model

    except Exception as e:
        print(f"Error loading model: {e}")

        return None


def open_video_capture(video_path):
    """
    Open video capture from the specified path.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: Video capture object, or None if an error occurred.
    """
    try:
        v_cap = cv2.VideoCapture(video_path)

        if not v_cap.isOpened():
            print("Error opening video capture.")

            return None

        return v_cap

    except Exception as e:
        print(f"Error opening video capture: {e}")

        return None


def process_objects(frame, model, object_counters, entered_ids):
    """
    Process detected objects, draw bounding boxes, and update counters.

    Args:
        frame (numpy.ndarray): Input frame.
        model (YOLO): YOLO model.
        object_counters (dict): Dictionary to log object counters by class.
        entered_ids (set): Set to record object IDs that have entered the rectangle.

    Returns:
        numpy.ndarray: Modified frame with bounding boxes and counters.
    """
    try:
        results = model.track(frame, persist = True, classes = [0, 2, 7], device = 'cuda')

        cv2.rectangle(frame, (RECT_X1, RECT_Y1), (RECT_X2, RECT_Y2), (255, 0, 0), 2)

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, class_idx, track_id in itools.zip_longest(boxes, class_ids, track_ids):
                x1, y1, x2, y2 = map(int, box[ : 4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                object_name = model.names[class_idx]

                cv2.putText(frame, f"ID: {track_id} {object_name}", (x1 - 5, y1 - 10), FONT, 0.5, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

                if (RECT_X1 < cx < RECT_X2 and RECT_Y1 < cy < RECT_Y2) and track_id not in entered_ids:
                    entered_ids.add(track_id)
                    object_counters[object_name] = object_counters.get(object_name, 0) + 1

        else:
            print("No objects detected in the current frame.")

        return frame

    except Exception as e:
        print(f"Error in object processing: {e}")

        return frame


def display_counters(frame, object_counters):
    """
    Display object counters on the frame.

    Args:
        frame (numpy.ndarray): Input frame.
        object_counters (dict): Dictionary of object counters.

    Returns:
        numpy.ndarray: Modified frame with counters displayed.
    """
    y_offset = COUNTER_TEXT_OFFSET

    for object_name, counter in object_counters.items():
        cv2.putText(frame, f"{object_name}: {counter}", (34, y_offset), FONT, 0.8, (0, 0, 0), 2)
        y_offset += COUNTER_TEXT_OFFSET

    return frame


def main():
    """
    Main function to run the object detection and counting.
    """
    model = load_model('yolo11m.pt')

    if model is None:
        return

    vid_cap = open_video_capture('videos/street.mp4')    # Enter here your path to your test video!!!

    if vid_cap is None:
        return

    object_counters = {}
    entered_ids = set()

    try:
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()

            if not ret:
                break

            frame = process_objects(frame, model, object_counters, entered_ids)
            frame = display_counters(frame, object_counters)

            cv2.imshow("YOLO11m Vehicles Counting", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("User's Ctrl+C detected.")

    finally:
        vid_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
