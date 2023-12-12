from taipy.gui import Gui
from tensorflow.keras import models
from PIL import Image
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input

class_names = {
    0: 'Not in Uniform',
    1: 'Uniform'
}


model = models.load_model("model.h5")

def predict_image(model, path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    
    # Preprocess input to match the training data preprocessing
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)

    # Get the class probabilities
    probs = model.predict(img_array)[0]

    # Get the class label with the highest probability
    top_prob = np.max(probs)
    top_pred = class_names[np.argmax(probs)]
    
    return top_prob, top_pred

    
content = ""
img_path = "placeholder_image.png"
prob = 0
pred = ""

index = """
<|text-center|
<|{"logo.png"}|image|width=25vw|>

<|{content}|file_selector|extensions=.png or .jpg|>
Select an image from your Device

<|{pred}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=25vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = f"Prediction: {top_pred}"
        state.img_path = var_val
    


app = Gui(page=index)

if __name__ == "__main__":
    app.run(use_reloader=True)
