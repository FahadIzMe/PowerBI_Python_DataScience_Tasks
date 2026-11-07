import gradio as gr
import pandas as pd
import joblib


model = joblib.load("car_price_linear_model.pkl")
model_rf = joblib.load("car_price_rf_model.pkl")
scaler = joblib.load("car_scaler.pkl")
brand_encoder = joblib.load("brand_encoder.pkl")
fuel_type_encoder = joblib.load("fuel_type_encoder.pkl")
transmission_encoder = joblib.load("transmission_encoder.pkl")

def predict_price(brand, model_year, engine_size, mileage, fuel_type, transmission, owner_count):
    
    df = pd.DataFrame([{
        "brand": brand,
        "model_year": model_year,
        "engine_size": engine_size,
        "mileage": mileage,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "owner_count": owner_count
    }])

    # Apply encoding
    df_encoded = df.copy()
    df_encoded["brand"] = brand_encoder.transform(df_encoded["brand"])
    df_encoded["fuel_type"] = fuel_type_encoder.transform(df_encoded["fuel_type"])
    df_encoded["transmission"] = transmission_encoder.transform(df_encoded["transmission"])
    
    # Apply scaling
    df_scaled = df_encoded.copy()
    df_scaled[["model_year", "engine_size", "mileage", "owner_count"]] = scaler.transform(df_scaled[["model_year", "engine_size", "mileage", "owner_count"]])

    # Predict
    price1 = model.predict(df_scaled)[0]
    price2 = model_rf.predict(df_scaled)[0]
    return f"Predicted Price_linear: Rs {price1:,.0f}\nPredicted Price_rf: Rs {price2:,.0f}"

# Gradio UI
ui = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(["Toyota","Honda","Suzuki","Hyundai","Kia","BMW","Audi","Mercedes","Ford","Nissan"], label="Brand"),
        gr.Number(label="Model Year"),
        gr.Number(label="Engine Size"),
        gr.Number(label="Mileage"),
        gr.Dropdown(["Petrol","Diesel","Hybrid"], label="Fuel Type"),
        gr.Dropdown(["Automatic","Manual"], label="Transmission"),
        gr.Number(label="Owner Count"),
    ],
    outputs=gr.Textbox(),
    title="Car Price Prediction System",
    description="Predict used car prices using your ML model."
)

ui.launch()
