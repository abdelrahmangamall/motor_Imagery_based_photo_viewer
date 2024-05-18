import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk


def move_image_right():
    global image_label
    canvas.move(image_label, 100, 0)  # Move 100 pixels to the right


def move_image_left():
    global image_label
    canvas.move(image_label, -100, 0)  # Move 100 pixels to the left
def load_data():
    global data_text
    file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "r") as file:
            data_text = file.read()
def domove():
    # Get user input or direction from somewhere
    import numpy as np
    import joblib
    data_text=[]
    file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, "r") as file:
            data_text = file.read()
    data_list = [float(x) for x in data_text.split()]
    print(data_list)

    import joblib

    # Load the model
    model = joblib.load("rf_model.pkl")
    print(model)
    # Now you can use the loaded model for prediction

    rf_model = joblib.load('rf_model.pkl')
    single_row_reshaped = np.array(data_list).reshape(1, -1)
    # Predict the label for the single row using the Random Forest model
    direction = rf_model.predict(single_row_reshaped)
    print("Predicted label:", direction)
    if direction == "right":
        move_image_right()
    elif direction == "left":
        move_image_left()


try:
    # Create the main window
    root = tk.Tk()
    root.title("Move Image")

    # Create a canvas
    canvas = tk.Canvas(root, width=400, height=300)
    canvas.pack()

    # Load the image
    image = Image.open("name.jpg")
    # Resize the image if necessary
    image = image.resize((100, 100))  # Adjust the size as needed
    # Convert the image to Tkinter format
    photo = ImageTk.PhotoImage(image)

    # Add an image to the canvas
    image_label = canvas.create_image(150, 100, anchor=tk.NW, image=photo)
    # Add buttons for data movement and data loading
    # Add a button for moving the image
    load_button = tk.Button(root, text="Do Move", command=domove)
    load_button.pack()

    # Run the Tkinter event loop
    root.mainloop()

except Exception as e:
    print("An error occurred:", e)
