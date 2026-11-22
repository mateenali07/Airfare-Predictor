# Airfare Predictor (Streamlit)

A supervised machine-learning app using a Random Forest Regressor to predict flight prices (INR).

## Setup

1. Run the Jupyter notebook to create:
   - `airfare_random_forest_pipeline.pkl`
   - `airfare_input_options.pkl`
2. Copy both `.pkl` files into the same directory as `app.py`.
3. Install: `python -m pip install -r requirements.txt`
4. Start: `python -m streamlit run app.py`

The app does not require the training CSV at prediction time.
