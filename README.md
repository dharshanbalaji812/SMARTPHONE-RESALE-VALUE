# Smartphone Resale Value Detection

A Flask + scikit-learn demo website that estimates a smartphone's second-hand resale price in INR.

## Requirements
- Python 3.10 or newer
- pip

## Run on macOS
1. Open Terminal and move into this folder:
   `cd smartphone-resale-prediction`
2. Create a virtual environment:
   `python3 -m venv venv`
3. Activate it:
   `source venv/bin/activate`
4. Install packages:
   `pip install -r requirements.txt`
5. (Optional) Regenerate the illustrative data and model:
   `python generate_dataset.py`
   `python train_model.py`
6. Start the website:
   `python app.py`
7. Open this address in your browser:
   `http://127.0.0.1:5000`

## Important dataset note
The included CSV is synthetic demo data created from an illustrative depreciation formula. It is useful for demonstrating the complete ML workflow, but it is NOT real market data and does not establish real-world prediction accuracy. For a credible project, replace it with a sufficiently large, cleaned dataset of actual sold/listed phones from your target market, then retrain the model.

## Main files
- `app.py`: Flask routes and prediction handling
- `train_model.py`: preprocessing, Random Forest training and evaluation
- `generate_dataset.py`: creates sample CSV data
- `smartphone_data.csv`: sample training dataset
- `models/resale_model.pkl`: trained model pipeline
- `templates/`: HTML pages
- `static/style.css`: website styling
