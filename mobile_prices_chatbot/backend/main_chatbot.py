import joblib
import pandas as pd
import os
import re

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- MODEL LOADING (Centralized and Robust) ---
REGRESSION_MODEL_PATH = r"models\mobile_price_regression_model.pkl"
CLASSIFICATION_MODEL_PATH = r"models\mobile_price_classification_model.pkl"

# Load Regression Model
try:
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    print(f"Regression model loaded successfully from: {REGRESSION_MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Regression model file not found at {REGRESSION_MODEL_PATH}.")
    print("Please ensure the 'models' directory exists and contains 'mobile_price_regression_model.pkl'.")
    class DummyRegressionModel: # Fallback dummy model
        def predict(self, X): return [0.0] * len(X)
    regression_model = DummyRegressionModel()
except Exception as e:
    print(f"Error loading regression model from {REGRESSION_MODEL_PATH}: {e}")
    class DummyRegressionModel: # Fallback dummy model
        def predict(self, X): return [0.0] * len(X)
    regression_model = DummyRegressionModel()

# Load Classification Model
try:
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    print(f"Classification model loaded successfully from: {CLASSIFICATION_MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Classification model file not found at {CLASSIFICATION_MODEL_PATH}.")
    print("Please ensure the 'models' directory exists and contains 'mobile_price_classification_model.pkl'.")
    class DummyClassificationModel: # Fallback dummy model
        def predict(self, X): return [0] * len(X)
    classification_model = DummyClassificationModel()
except Exception as e:
    print(f"Error loading classification model from {CLASSIFICATION_MODEL_PATH}: {e}")
    class DummyClassificationModel: # Fallback dummy model
        def predict(self, X): return [0] * len(X)
    classification_model = DummyClassificationModel()

# --- TOOL FUNCTIONS ---

def predict_mobile_price_regression_tool(input_string: str) -> str:
    """
    Predicts the EXACT mobile phone price based on 11 specific numerical features.
    The input_string MUST be a comma-separated string of 11 values IN THIS ORDER:
    weight (float), resoloution (float), ppi (int), cpu_core (int), cpu_freq (float),
    internal_mem (int), ram (int), RearCam (int), Front_Cam (int), battery (int), thickness (float).
    Example: "180.5, 5.8, 400, 8, 2.8, 128, 8, 64, 16, 4500, 8.5"
    """
    try:
        # Use regex to find all numbers, including floats
        params = [float(s) for s in re.findall(r'\d+\.?\d*', input_string)]

        if len(params) != 11:
            return (f"Error: Invalid input for exact price prediction. Expected 11 numerical values, "
                    f"but received {len(params)}. Please ensure you provide all 11 values in the correct order.")

        weight, resoloution, ppi, cpu_core, cpu_freq, internal_mem, ram, RearCam, Front_Cam, battery, thickness = params

        X_new = pd.DataFrame([{
            "weight": weight, "resoloution": resoloution, "ppi": ppi, "cpu core": cpu_core,
            "cpu freq": cpu_freq, "internal mem": internal_mem, "ram": ram,
            "RearCam": RearCam, "Front_Cam": Front_Cam, "battery": battery, "thickness": thickness
        }])

        pred = regression_model.predict(X_new)
        return f"The predicted exact mobile price is: ${pred[0]:.2f}"
    except Exception as e:
        return f"An error occurred during exact price prediction: {e}. Please ensure the input format is correct."

def predict_mobile_price_classification_tool(input_string: str) -> str:
    """
    Classifies the mobile phone price into a range (low cost, medium cost, high cost, very high cost)
    based on 20 specific numerical features.
    The input_string MUST be a comma-separated string of 20 values IN THIS ORDER:
    battery_power (int), blue (int), clock_speed (float), dual_sim (int), fc (int),
    four_g (int), int_memory (int), m_dep (float), mobile_wt (int), n_cores (int),
    pc (int), px_height (int), px_width (int), ram (int), sc_h (int), sc_w (int),
    talk_time (int), three_g (int), touch_screen (int), wifi (int).
    Example: "2000, 1, 2.5, 0, 8, 1, 128, 0.7, 160, 8, 20, 1920, 1080, 4000, 15, 7, 18, 1, 1, 1"
    """
    try:
        # Use regex to find all numbers, including floats
        params = [float(s) for s in re.findall(r'\d+\.?\d*', input_string)]

        if len(params) != 20:
            return (f"Error: Invalid input for price range prediction. Expected 20 numerical values, "
                    f"but received {len(params)}. Please ensure you provide all 20 values in the correct order.")

        (battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt,
         n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi) = params

        X_new = pd.DataFrame([{
            "battery_power": int(battery_power), "blue": int(blue), "clock_speed": clock_speed,
            "dual_sim": int(dual_sim), "fc": int(fc), "four_g": int(four_g),
            "int_memory": int(int_memory), "m_dep": m_dep, "mobile_wt": int(mobile_wt),
            "n_cores": int(n_cores), "pc": int(pc), "px_height": int(px_height),
            "px_width": int(px_width), "ram": int(ram), "sc_h": int(sc_h), "sc_w": int(sc_w),
            "talk_time": int(talk_time), "three_g": int(three_g), "touch_screen": int(touch_screen),
            "wifi": int(wifi)
        }])

        prediction_raw = classification_model.predict(X_new)[0]

        price_ranges = {
            0: "low cost", 1: "medium cost", 2: "high cost", 3: "very high cost"
        }
        predicted_range = price_ranges.get(prediction_raw, "an unknown price range")
        return f"The predicted mobile price range is: {predicted_range}"

    except Exception as e:
        return f"An error occurred during price range prediction: {e}. Please ensure the input format is correct."

# --- CHATBOT CREATION ---

def create_chatbot_with_memory_and_tools():
    """
    Creates a LangChain chatbot with ConversationBufferMemory and integrated tools.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
    )

    tools = [
        Tool(
            name="MobilePricePredictorExact",
            func=predict_mobile_price_regression_tool,
            description="""This tool predicts the **EXACT price** of a mobile phone.
            Input: A comma-separated string of **11 numerical values only**, in this order:
            weight, resoloution, ppi, cpu_core, cpu_freq, internal_mem, ram, RearCam, Front_Cam, battery, thickness.
            Example: "180.5, 5.8, 400, 8, 2.8, 128, 8, 64, 16, 4500, 8.5"
            """
        ),
        Tool(
            name="MobilePricePredictorRange",
            func=predict_mobile_price_classification_tool,
            description="""This tool classifies the mobile phone price into a **range** (low/medium/high/very high cost).
            Input: A comma-separated string of **20 numerical values only**, in this order:
            battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores,
            pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi.
            Example: "2000, 1, 2.5, 0, 8, 1, 128, 0.7, 160, 8, 20, 1920, 1080, 4000, 15, 7, 18, 1, 1, 1"
            """
        )
    ]

    # --- REVISED PROMPT TEMPLATE FOR EXPLICIT EXTRACTION AND FLOW ---
    prompt_template = """
    You are a helpful and intelligent AI assistant specializing in mobile phone price prediction.
    You have access to two distinct tools: MobilePricePredictorExact and MobilePricePredictorRange. You can also engage in general conversation.

    Here are your tools and their precise input requirements:

    {tools}

    Follow these strict rules for generating your responses:

    1.  **General Conversation:** If the user's question is a greeting, a general query, or clearly does NOT involve mobile phone specifications for prediction, respond directly with a **Final Answer**. Do NOT use any tools or attempt to format an Action.

    2.  **Tool Usage - When to Plan:**
        * If the user expresses intent for an **"exact price"** prediction and provides specifications, you should plan to use `MobilePricePredictorExact`.
        * If the user expresses intent for a **"price range"** or **"cost category"** prediction and provides specifications, you should plan to use `MobilePricePredictorRange`.

    3.  **Tool Input Extraction - CRITICAL:**
        * When using a tool, your `Action Input` MUST be a **clean, comma-separated string of ONLY the numerical values**.
        * You must parse the user's input to extract *only* the numbers and order them correctly, removing any descriptive text (like "weight =", "ram:", etc.).

    4.  **Handling Incomplete Tool Input OR Initial Tool Query Without Data:**
        * If you identify that a tool is needed (e.g., user asks "predict price" or "I have specs") but the user has **NOT provided ALL the necessary numerical values** for that specific tool *in the current turn*, **do NOT use the tool.**
        * Instead, respond with a **Final Answer** that politely explains which specific tool is intended (if clear), states how many parameters are missing, and/or lists the *full set of parameters expected* for that tool, asking the user to provide them.
        * **Important:** In this scenario (asking for more info), your output MUST ONLY contain a `Final Answer`. Do NOT include `Action:` or `Action Input:` lines.

    5.  **Direct Tool Invocation:**
        * If a tool is needed AND **ALL required numerical parameters are present** in the user's current input, then proceed to call the tool.
        * In this case, your output should follow the `Thought` -> `Action` -> `Action Input` -> `Observation` -> `Thought` -> `Final Answer` format.

    Your thinking process should strictly follow this format:

    Question: the input question you must answer
    Thought:
        1.  Analyze the user's intent: Is it general conversation, or a request for price prediction?
        2.  If general conversation: Formulate a direct Final Answer.
        3.  If price prediction:
            a.  Identify if an "exact price" or "price range" is requested (or infer based on number of parameters if available).
            b.  Attempt to extract all numerical parameters from the user's input.
            c.  **Check if all necessary parameters are present for the identified tool.**
                i.  If parameters are missing (or if user only indicated intent without providing any specs yet): Formulate a polite **Final Answer** asking for the complete information. Do NOT include `Action` or `Action Input`.
                ii. If all parameters are present: Proceed to define `Action` and `Action Input` to call the tool.
    Action: the action to take (must be one of [{tool_names}]). This line and 'Action Input:' should ONLY appear if you are calling a tool.
    Action Input: the extracted, clean comma-separated numerical values for the tool. This line should ONLY appear if you are calling a tool.
    Observation: the result of the action (only if an Action was taken).
    Thought: I now know the final answer based on the Observation.
    Final Answer: the final answer to the original input question.

    Begin!

    {chat_history}
    Question: {input}
    Thought:{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")
    )
    return agent_executor

# --- CHAT INTERFACE ---

def chat_with_bot(chatbot):
    """
    Starts an interactive chat session with the bot.
    """
    print("Chatbot with memory and tools initialized. Type 'exit' to end the conversation.")
    print("\n--- How to Use Me ---")
    print("1. **To predict an EXACT price:** Ask 'What is the exact price of a phone with...' followed by the 11 specifications.")
    print("   You can say things like 'weight=180.5, resolution=5.8, ...' or just list the numbers: '180.5, 5.8, ...'")
    print("   Example Full Query: 'What is the exact price of a phone with weight 180.5, resolution 5.8, ppi 400, 8 cpu cores, 2.8 cpu frequency, 128 internal memory, 8 ram, 64MP rear cam, 16MP front cam, 4500mAh battery, and 8.5mm thickness?'")
    print("\n2. **To predict a PRICE RANGE:** Ask 'What is the price range for a phone with...' followed by the 20 specifications.")
    print("   You can say things like 'battery_power=2000, blue=1, ...' or just list the numbers: '2000, 1, ...'")
    print("   Example Full Query: 'What is the price range for a phone with battery power 2000, bluetooth 1, clock speed 2.5, dual sim 0, front camera 8, 4G 1, internal memory 128, mobile depth 0.7, mobile weight 160, 8 cores, primary camera 20, pixel height 1920, pixel width 1080, ram 4000, screen height 15, screen width 7, talk time 18, 3G 1, touch screen 1, and wifi 1?'")
    print("\n3. **For general conversation:** Just type your message (e.g., 'Hi', 'Tell me a joke.').")
    print("\nType 'exit' to end the conversation at any time.")
    print("--------------------------\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        try:
            response = chatbot.invoke({"input": user_input})
            print(f"Chatbot: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}. This might be due to an issue with the Groq API or an unexpected LLM response format. Please try again or check your API key/network connection.")

if __name__ == "__main__":
    chatbot = create_chatbot_with_memory_and_tools()
    chat_with_bot(chatbot)