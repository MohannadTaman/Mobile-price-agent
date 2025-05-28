import joblib
import pandas as pd
import os

model_path = r"models\mobile_price_regression_model.pkl"
model = joblib.load(model_path)

def predict_mobile_price_regression(
    weight: float, resoloution: float, ppi: int, cpu_core: int, cpu_freq: float,
    internal_mem: int, ram: int, RearCam: int, Front_Cam: int, battery: int, thickness: float
):
    X_new = pd.DataFrame([{
        "weight": weight,
        "resoloution": resoloution,
        "ppi": ppi,
        "cpu core": cpu_core,
        "cpu freq": cpu_freq,
        "internal mem": internal_mem,
        "ram": ram,
        "RearCam": RearCam,
        "Front_Cam": Front_Cam,
        "battery": battery,
        "thickness": thickness
    }])

    pred = model.predict(X_new)
    return f"The predicted mobile price is: ${pred[0]:.2f}"

if __name__ == "__main__":
    print(predict_mobile_price_regression(180.5, 5.8, 400, 8, 2.8, 128, 8, 64, 16, 4500, 8.5))
    # print(predict_mobile_price_regression(135.5, 5.3, 424, 8, 1.350, 16, 3, 13, 8, 2160, 7.4))


#Test agent with: What is the exact price for a phone with these specs: 180.5, 5.8, 400, 8, 2.8, 128, 8, 64, 16, 4500, 8.5

#Test agent with: What is the exact price for a phone with these specs: 135.5, 5.3, 424, 8, 1.350, 16, 3, 13, 8, 2160, 7.4