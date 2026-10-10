import argparse
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO weights (n=fast, s/m/l=more accurate)")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--classes", nargs="*", default=None, help="only track these class names")
    p.add_argument("--trail", type=int, default=30, help="trail length in frames")
    return p.parse_args()


def color_for_id(track_id: int):
    import colorsys
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    names = model.names

    class_ids = None
    if args.classes:
        wanted = {c.lower() for c in args.classes}
        class_ids = [i for i, n in names.items() if n.lower() in wanted]
        if not class_ids:
            raise SystemExit(f"No matching classes. Available: {list(names.values())}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    trails = defaultdict(lambda: deque(maxlen=args.trail))
    seen_ids = set()
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            conf=args.conf,
            classes=class_ids,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, cid, conf in zip(boxes, ids, clss, confs):
                seen_ids.add(tid)
                color = color_for_id(tid)
                label = f"#{tid} {names[cid]} {conf:.0%}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trails[tid].append((cx, cy))
                pts = trails[tid]
                for i in range(1, len(pts)):
                    thickness = max(1, int(3 * i / len(pts)))
                    cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}  Unique tracked: {len(seen_ids)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Camera Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print("Saved", fname)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()