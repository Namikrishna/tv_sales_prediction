from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = "model/tv_sales_model.pkl"
scaler_path = "model/scaler.pkl"

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    print("Error: Model or scaler file not found. Please run 'train_model.py' first.")
    exit()

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            tv_budget = float(request.form['tv_budget'])
            if tv_budget < 0:
                error = "TV Advertising Budget cannot be negative!"
            else:
                tv_budget_array = np.array([[tv_budget]])
                tv_budget_scaled = scaler.transform(tv_budget_array)
                prediction = model.predict(tv_budget_scaled)[0]
                prediction = f"${prediction:,.2f}"
        except ValueError:
            error = "Please enter a valid number!"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)