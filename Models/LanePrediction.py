"""Lane detection + lane position check for a moving car.
"""
import os
import cv2
import numpy as np

SOURCE = "road4.mp4"
SAVE_OUTPUT = True           
LANE_WIDTH_M = 3.7          
DRIFT_LIMIT_M = 0.35        
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def region_of_interest(edges):
   
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[(int(0.05 * w), h), (int(0.45 * w), int(0.60 * h)),
                      (int(0.55 * w), int(0.60 * h)), (int(0.95 * w), h)]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask)


def make_line(h, fit):
    
    slope, intercept = fit
    y1, y2 = h, int(h * 0.60)
    x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
    return x1, y1, x2, y2


def average_lanes(lines, h):
   
    left, right = [], []
    if lines is None:
        return None, None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x1 == x2:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if abs(slope) < 0.4:              # ignore near-horizontal lines
            continue
        (left if slope < 0 else right).append((slope, intercept))
    l = make_line(h, np.mean(left, axis=0)) if left else None
    r = make_line(h, np.mean(right, axis=0)) if right else None
    return l, r


def process_frame(frame):
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=100)
    left, right = average_lanes(lines, h)

    overlay = np.zeros_like(frame)
    status, color = "Lane not detected", (0, 0, 255)

    for ln in (left, right):
        if ln is not None:
            cv2.line(overlay, (ln[0], ln[1]), (ln[2], ln[3]), (0, 255, 0), 8)

    if left is not None and right is not None:
        # fill the lane area
        pts = np.array([[left[0], left[1]], [left[2], left[3]],
                        [right[2], right[3]], [right[0], right[1]]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 120, 0))

        lane_center = (left[0] + right[0]) / 2
        lane_px = max(right[0] - left[0], 1)
        offset_m = (w / 2 - lane_center) * (LANE_WIDTH_M / lane_px)   # + means car is right of center

        if abs(offset_m) <= DRIFT_LIMIT_M:
            status, color = f"In lane - centered ({offset_m:+.2f} m)", (0, 200, 0)
        elif offset_m > 0:
            status, color = f"WARNING: drifting RIGHT ({offset_m:+.2f} m)", (0, 140, 255)
        else:
            status, color = f"WARNING: drifting LEFT ({offset_m:+.2f} m)", (0, 140, 255)

        cv2.line(overlay, (int(lane_center), h), (int(lane_center), h - 40), (255, 255, 0), 4)  # lane center
        cv2.line(overlay, (w // 2, h), (w // 2, h - 40), (0, 0, 255), 4)                        # car center
    elif left is not None or right is not None:
        status, color = "Only one lane line detected", (0, 200, 255)

    out = cv2.addWeighted(frame, 0.85, overlay, 0.6, 0)
    cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(out, status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out, status


def main():
    is_image = isinstance(SOURCE, str) and SOURCE.lower().endswith((".jpg", ".jpeg", ".png"))

    if is_image:
        frame = cv2.imread(SOURCE)
        if frame is None:
            raise FileNotFoundError(f"Could not open {SOURCE}")
        out, status = process_frame(frame)
        print(status)
        if SAVE_OUTPUT:
            cv2.imwrite("lane_output.jpg", out)
        cv2.imshow("Lane detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, status = process_frame(frame)

        if SAVE_OUTPUT and writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter("lane_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer is not None:
            writer.write(out)

        cv2.imshow("Lane detection (press q to quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()