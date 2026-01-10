"""
Provides a GUI for manually annotating facial landmarks (e.g., eyes, nose) on face images.
Creates the ground truth region mapping required for the quantitative regional analysis of LRP heatmaps.
"""
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import csv
from facesim3d import local_paths
import os

# =====================================================
# ==== CONFIGURATION ====
REGIONS = [
    "mouth", "space between eyes",
    "nose_bridge", "nose_tip",
    "left_eye", "right_eye",
    "left_eyebrow", "right_eyebrow",
    "forehead", "left_cheek", "right_cheek",
    "left_ear", "right_ear", "chin", "contour"
]
SAVE_DIR = local_paths.DIR_REGION_ANALYSIS_RESULTS
SAVE_PATH = os.path.join(SAVE_DIR, "landmark_annotations.csv")
# =====================================================

def start_annotation(image, points):
    """Launches the annotation GUI"""
    annotations = {}
    total_points = len(points)
    current_index = {"i": 0}

    # ----- Tkinter setup -----
    root = tk.Tk()
    root.title("Facial Landmark Region Annotator")

    # Resize image for comfortable viewing
    display_width = 400
    scale = display_width / image.shape[1]
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    scaled_points = [(int(x * scale), int(y * scale)) for (x, y) in points]

    # Convert to Tk image
    def get_image_with_point(idx):
        temp = resized.copy()
        x, y = scaled_points[idx]
        cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(temp, f"#{idx}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        img_rgb = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(img_rgb))

    img_label = tk.Label(root)
    status_label = tk.Label(root, text="", font=("Helvetica", 12))
    img_label.pack(pady=5)
    status_label.pack(pady=2)

    status_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"))
    status_label.pack(pady=5)

    # ----- Button Handlers -----
    def assign_region(region_name):
        i = current_index["i"]
        annotations[i] = region_name
        current_index["i"] += 1

        if current_index["i"] < total_points:
            update_display()
        else:
            save_annotations()
            messagebox.showinfo("Done", f"All landmarks annotated and saved to {SAVE_PATH}")
            root.destroy()

    def update_display():
        i = current_index["i"]
        img_tk = get_image_with_point(i)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        status_label.config(text=f"Annotating landmark {i + 1}/{total_points}")

    def save_annotations():
        with open(SAVE_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["landmark_index", "region"])
            for k, v in annotations.items():
                writer.writerow([k, v])

    def undo_last():
        """Undo last annotation and go back one landmark."""
        if current_index["i"] > 0:
            current_index["i"] -= 1
            annotations.pop(current_index["i"], None)
            update_display()
        else:
            messagebox.showinfo("Info", "Nothing to undo.")

    def save_and_exit():
        """Save current annotations and close GUI."""
        save_annotations()
        messagebox.showinfo("Saved", f"Progress saved to {SAVE_PATH}")
        root.destroy()

    # ----- Region Buttons -----
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=15)

    # Larger buttons, arranged in a grid
    for i, region in enumerate(REGIONS):
        btn = tk.Button(
            btn_frame,
            text=region,
            width=15,  # larger button width
            height=2,  # taller buttons
            font=("Helvetica", 10, "bold"),
            bg="#e0e0e0",
            command=lambda r=region: assign_region(r)
        )
        btn.grid(row=i // 5, column=i % 5, padx=6, pady=4)

    # ----- Extra control buttons -----
    control_frame = tk.Frame(root)
    control_frame.pack(pady=8)

    undo_btn = tk.Button(
        control_frame,
        text="⟲ Undo Last",
        width=12,
        height=1,
        font=("Helvetica", 10, "bold"),
        bg="#ffcccb",
        command=lambda: undo_last()
    )
    undo_btn.pack(side=tk.LEFT, padx=10)

    save_btn = tk.Button(
        control_frame,
        text="Save & Exit",
        width=14,
        height=1,
        font=("Helvetica", 10, "bold"),
        bg="#d0f0c0",
        command=lambda: save_and_exit()
    )
    save_btn.pack(side=tk.LEFT, padx=10)

    # ----- Start -----
    update_display()
    root.mainloop()


# =====================================================
# ==== EXAMPLE USAGE ====
if __name__ == "__main__":
    import mediapipe as mp
    import numpy as np

    import sys, mediapipe
    print("Python:", sys.version)
    print("MediaPipe:", mediapipe.__version__)
    print("Path:", mediapipe.__file__)

    image_folder = local_paths.DIR_FRONTAL_VIEW_HEADS

    image_path = image_folder + "/head-001_frontal.png"  # check afterwards if this works for all IDs
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise ValueError("No face detected.")

    landmarks = results.multi_face_landmarks[0].landmark
    points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

    start_annotation(image, points)
