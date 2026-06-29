import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/traffic_sign_model.h5")

# Class names (Update if your dataset uses different classes)
classes = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End of Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Overtaking",
    10: "No Overtaking for Trucks",
    11: "Right of Way",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve Left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice",
    31: "Wild Animals Crossing",
    32: "End of Restrictions",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout",
    41: "End of No Overtaking",
    42: "End of No Overtaking by Trucks"
}

root = tk.Tk()
root.title("Traffic Sign Detection and Recognition System")
root.geometry("700x600")

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16), fg="blue")
result_label.pack(pady=10)

def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if file_path:
        img = Image.open(file_path)
        display = img.resize((300, 300))
        photo = ImageTk.PhotoImage(display)

        image_label.config(image=photo)
        image_label.image = photo

        img = img.resize((64, 64))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        result_label.config(text="Prediction: " + classes[class_index])

upload_btn = tk.Button(
    root,
    text="Browse Image",
    command=upload_image,
    font=("Arial", 14),
    bg="green",
    fg="white"
)
upload_btn.pack(pady=20)

root.mainloop()