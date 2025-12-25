import cv2
import os

class Annotator:
    def __init__(self, fps, frame_size):
        os.makedirs("outputs", exist_ok=True)
        self.writer = cv2.VideoWriter(
            "outputs/annotated_video.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            fps,
            frame_size
        )

    def draw_and_write(self, frame, tracks, count):
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.writer.write(frame)

    def close(self):
        self.writer.release()
# End of bird-count-weight-estimation/core/annotator.py
