import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load the trained model
@st.cache_resource # This keeps the model in memory so it doesn't reload every time you move a slider
def load_model():
    with open('gold_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. Build the Web Interface
st.title("🪙 Gold Price Prediction Dashboard")
st.write("Adjust the economic indicators below to see the AI's predicted price for Gold (GLD).")
st.markdown("---")

# 3. Create the input sliders in columns for a clean design
col1, col2 = st.columns(2)

with col1:
    spx = st.slider("S&P 500 Index (SPX)", min_value=1000.0, max_value=3000.0, value=1447.16)
    uso = st.slider("Crude Oil Price (USO)", min_value=10.0, max_value=120.0, value=78.47)

with col2:
    slv = st.slider("Silver Price (SLV)", min_value=8.0, max_value=50.0, value=15.18)
    eur_usd = st.slider("EUR/USD Exchange Rate", min_value=1.00, max_value=1.60, value=1.47)

st.markdown("---")

# 4. Format the inputs to match exactly what the model expects
input_data = pd.DataFrame({
    'SPX': [spx],
    'USO': [uso],
    'SLV': [slv],
    'EUR/USD': [eur_usd]
})

# 5. Make the Prediction
if st.button("Predict Gold Price", type="primary"):
    # The model outputs a list, so we grab the first item [0]
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Predicted Gold (GLD) Price: ${prediction:.2f}")
    st.caption("Random Forest Model Accuracy: ~98%")