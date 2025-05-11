
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)  # Entry point of the app

# Route for landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction form
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Render form on GET request
    else:
        try:
            # Retrieve form data safely
            SPX = request.form.get('SPX')
            GLD = request.form.get('GLD')
            USO = request.form.get('USO')
            SLV = request.form.get('SLV')

            # Validate all inputs exist
            if None in [SPX, GLD, USO, SLV]:
                return "Error: All input fields must be filled.", 400

            # Convert to float safely
            data = CustomData(
                SPX=float(SPX),
                GLD=float(GLD),
                USO=float(USO),
                SLV=float(SLV)
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

        except Exception as e:
            return f"An error occurred: {e}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
