from flask import Flask, request, render_template
from src.pipeline.predictive_pipeline import ModelPredictor

app = Flask(__name__)
model_predictor = ModelPredictor()

@app.route('/model_predictor', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        try:
            prediction = model_predictor.predict(input_text)
            print("Prediction:", prediction)  # Debugging statement
            return render_template('index.html', input_text=input_text, prediction=prediction)
        except Exception as e:
            print("Error:", e)  # Debugging statement
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
