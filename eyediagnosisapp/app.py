from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load pre-trained models
models = {
    "conjunctivitis": tf.keras.models.load_model('models/conjunctivitis_model.h5'),
    "dry_eye": tf.keras.models.load_model('models/dry_eye_model.h5'),
    "keratoconus": tf.keras.models.load_model('models/keratoconus_model.h5'),
    "cataract": tf.keras.models.load_model('models/cataract_model.h5'),
    "stye_chalazion": tf.keras.models.load_model('models/stye_chalazion_model.h5'),
    "blepharitis": tf.keras.models.load_model('models/blepharitis_model.h5'),
    "ocular_surface": tf.keras.models.load_model('models/ocular_surface_model.h5'),
    "allergic_reaction": tf.keras.models.load_model('models/allergic_reaction_model.h5')
}


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming the model expects 224x224 input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = "uploaded_image.jpg"
            file.save(file_path)
            img = prepare_image(file_path)

            # Make predictions
            predictions = {}
            for condition, model in models.items():
                prediction = model.predict(img)
                predictions[condition] = prediction[0][0]  # Assuming binary classification with sigmoid output

            return render_template("index.html", predictions=predictions)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
