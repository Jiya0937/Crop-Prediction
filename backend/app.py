from flask import Flask, render_template, request
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")

flask_app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR
)

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():

    float_features = [float(x) for x in request.form.values()]
    features = np.array([float_features])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text="The Predicted Crop is {}".format(prediction[0])
    )


if __name__ == "__main__":
    flask_app.run(debug=True)