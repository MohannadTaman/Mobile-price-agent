import joblib
import pandas as pd
import os

model_path = r"models\mobile_price_classification_model.pkl"
model = joblib.load(model_path)

def predict_mobile_price_classification(
    battery_power: int,
    blue: int,
    clock_speed: float,
    dual_sim: int,
    fc: int,
    four_g: int,
    int_memory: int,
    m_dep: float,
    mobile_wt: int,
    n_cores: int,
    pc: int,
    px_height: int,
    px_width: int,
    ram: int,
    sc_h: int,
    sc_w: int,
    talk_time: int,
    three_g: int,
    touch_screen: int,
    wifi: int
):
    X_new = pd.DataFrame([{
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi
    }])

    pred = model.predict(X_new)[0]
    price_ranges = {
        0: "low cost",
        1: "medium cost",
        2: "high cost",
        3: "very high cost"
    }
    return f"The predicted mobile price range is: {price_ranges.get(pred, 'unknown range')}"

if __name__ == "__main__":
    print(predict_mobile_price_classification(2000, 1, 2.5, 0, 8, 1, 128, 0.7, 160, 8, 20, 1920, 1080, 4000, 15, 7, 18, 1, 1, 1))
    # print(predict_mobile_price_classification(1500, 0, 1.8, 1, 5, 0, 32, 0.9, 140, 4, 8, 800, 720, 1500, 12, 6, 10, 0, 0, 0))

#Test agent with :What is the price range for a phone with specs: 2000, 1, 2.5, 0, 8, 1, 128, 0.7, 160, 8, 20, 1920, 1080, 4000, 15, 7, 18, 1, 1, 1
#Test agent with :What is the price range for a phone with specs: 21500, 0, 1.8, 1, 5, 0, 32, 0.9, 140, 4, 8, 800, 720, 1500, 12, 6, 10, 0, 0, 0