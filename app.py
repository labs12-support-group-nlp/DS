from flask import Flask, request, jsonify
from .predict import clean


app = Flask(__name__)

@app.route('/', methods=['POST','OPTIONS'])
def getText():
    text = request.get_json(force=True)
    return jsonify({'received': res_text,
                    "about":"Hey there!",
                    "group1": "Emotional issues support group",
                    "group2": "Mental issues support group",
                    "group3": "Mood issues support group",
                    "group4": "Depression issues support group",
                    "group5": "All kind of issues support group"})


# @app.route('/predict', methods=['GET'])
# def predict():
