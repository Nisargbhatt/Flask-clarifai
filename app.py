import base64

from flask import Flask, jsonify, request, render_template
from clarifai.rest import ClarifaiApp

app = Flask(__name__)
clarifai = ClarifaiApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image = base64.b64encode(request.files.get('image').read())
    model = clarifai.models.get('general-v1.3')
    response = model.predict_by_base64(image)
    return render_template('display_output.html', clarifai_response=response)

if __name__ == '__main__':
    app.run(debug=True)
