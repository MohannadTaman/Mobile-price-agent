
# Mobile Price Chatbot

---

This repository contains a Python-based chatbot designed to predict mobile phone prices. It leverages machine learning models (Regression and Classification) to provide either an **exact price** or a **price range** based on various phone specifications. The chatbot is built using LangChain for conversational capabilities and tool integration, allowing for natural language interaction.

## Features

* **Exact Price Prediction:** Predicts a precise monetary value for a mobile phone based on 11 key specifications.
* **Price Range Classification:** Categorizes a mobile phone's price into "low cost," "medium cost," "high cost," or "very high cost" based on 20 detailed specifications.
* **Conversational Interface:** Interact with the chatbot naturally, asking for predictions or engaging in general conversation.
* **Memory:** The chatbot remembers past interactions within a session, making follow-up questions more seamless.
* **Robust Input Handling:** Designed to guide users when specifications are missing or the input format is incorrect.

## Getting Started

Follow these steps to set up and run the Mobile Price Chatbot on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* **pip** (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/MohannadTaman/Mobile-price-agent.git](https://github.com/MohannadTaman/Mobile-price-agent.git)
    cd Mobile-price-agent
    ```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the `Mobile-price-agent` directory (at the same level as `backend` and `models` folders) and add your Groq API key:

    ```
    GROQ_API_KEY='your_groq_api_key_here'
    ```

    You can obtain a Groq API key from [Groq Console](https://console.groq.com/keys).

### Model Files

The chatbot relies on pre-trained machine learning models. These models are expected to be located in the `models/` directory.

* `mobile_price_regression_model.pkl` (for exact price prediction)
* `mobile_price_classification_model.pkl` (for price range prediction)

Ensure these `.pkl` files are present in the `models/` folder. If they are missing, the chatbot will print an error and use dummy models, which will not provide accurate predictions.

---

## How to Run the Chatbot

After completing the installation, you have two ways to run the chatbot:

1.  **Run the core backend chatbot (console-based):**

    Navigate to the `backend` directory and run the `main_chatbot.py` script:

    ```bash
    cd backend
    python main_chatbot.py
    ```

    The chatbot will initialize in your console and provide instructions on how to interact with it.

2.  **Run the Streamlit frontend (web-based):**

    Navigate to the root directory of your project (`Mobile-price-agent/`) and run the Streamlit application:

    ```bash
    streamlit run frontend/app.py
    ```

    This will open the chatbot in your web browser, providing a graphical user interface.

---

## How to Use the Chatbot

Once the chatbot is running, you can interact with it in the console or web interface.

### 1. To Predict an **EXACT Price**

Ask 'What is the exact price of a phone with...' followed by the 11 specifications. You can list the specifications as `key=value` pairs or simply provide the numbers in order.

* **Required 11 Specifications (in order):** `weight`, `resolution`, `ppi`, `cpu_core`, `cpu_freq`, `internal_mem`, `ram`, `RearCam`, `Front_Cam`, `battery`, `thickness`.

* **Example Full Query:**
    `What is the exact price of a phone with weight 180.5, resolution 5.8, ppi 400, 8 cpu cores, 2.8 cpu frequency, 128 internal memory, 8 ram, 64MP rear cam, 16MP front cam, 4500mAh battery, and 8.5mm thickness?`

* **Shortened Example:**
    `180.5, 5.8, 400, 8, 2.8, 128, 8, 64, 16, 4500, 8.5`

### 2. To Predict a **PRICE RANGE**

Ask 'What is the price range for a phone with...' followed by the 20 specifications. You can use `key=value` pairs or just list the numbers.

* **Required 20 Specifications (in order):** `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`.

* **Example Full Query:**
    `What is the price range for a phone with battery power 2000, bluetooth 1, clock speed 2.5, dual sim 0, front camera 8, 4G 1, internal memory 128, mobile depth 0.7, mobile weight 160, 8 cores, primary camera 20, pixel height 1920, pixel width 1080, ram 4000, screen height 15, screen width 7, talk time 18, 3G 1, touch screen 1, and wifi 1?`

* **Shortened Example:**
    `2000, 1, 2.5, 0, 8, 1, 128, 0.7, 160, 8, 20, 1920, 1080, 4000, 15, 7, 18, 1, 1, 1`

### 3. For **General Conversation**

Just type your message (e.g., 'Hi', 'Tell me a joke.'). The chatbot will engage in a normal conversation without using its prediction tools.

### Ending the Conversation

Type `exit` at any time to end the chatbot session.

---

## Project Structure
