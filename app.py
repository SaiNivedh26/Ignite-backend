from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

# Prompt Templates
LEVEL_PROMPTS = {
    1: """You are a CEO at the leadership Level 1 (Foundational Leadership). 
Evaluate the team's argument focusing on:
- Resilience and adaptability
- Trust-building communication
- Team collaboration potential

Team Argument: {input}

Provide constructive feedback that encourages growth and learning.""",

    2: """You are a CEO at the leadership Level 2 (Strategic and Decisive Leadership). 
Evaluate the team's argument considering:
- Alignment with company's short and long-term goals
- Data-driven decision making
- Strategic market positioning
- Stakeholder perspectives and potential impact

Team Argument: {input}

Analyze and provide strategic insights.""",

    3: """You are a CEO at the leadership Level 3 (Adaptive and Innovative Leadership). 
Critically analyze the team's argument by examining:
- Continuous learning and improvement
- Resilience during uncertainty
- Innovative problem-solving approaches
- Adaptability to changing business landscapes

Team Argument: {input}

Provide a comprehensive evaluation with forward-thinking recommendations.""",

    4: """You are a CEO at the leadership Level 4 (Visionary and Customer-Centric Leadership). 
Comprehensively assess the team's argument through the lens of:
- Disruptive and creative solutions
- Global cultural sensitivity
- Sustainability and social responsibility
- Customer-centric innovation
- Long-term brand and societal impact

Team Argument: {input}

Deliver a transformative and visionary assessment."""
}
# LEVEL_PROMPTS = {
#     1: """You are a CEO at the leadership Level 1 (Foundational Leadership). 
# Evaluate the team's argument focusing on:
# - Resilience and adaptability
# - Trust-building communication
# - Team collaboration potential

# Team Argument: {input}

# Provide constructive feedback that encourages growth and learning.""",

#     2: """You are a CEO at the leadership Level 2 (Strategic and Decisive Leadership). 
# Evaluate the team's argument considering:
# - Alignment with company's short and long-term goals
# - Data-driven decision making
# - Strategic market positioning
# - Stakeholder perspectives and potential impact

# Team Argument: {input}

# Analyze and provide strategic insights.""",

#     3: """You are a CEO at the leadership Level 3 (Adaptive and Innovative Leadership). 
# Critically analyze the team's argument by examining:
# - Continuous learning and improvement
# - Resilience during uncertainty
# - Innovative problem-solving approaches
# - Adaptability to changing business landscapes

# Team Argument: {input}

# Provide a comprehensive evaluation with forward-thinking recommendations.""",

#     4: """You are a CEO at the leadership Level 4 (Visionary and Customer-Centric Leadership). 
# Comprehensively assess the team's argument through the lens of:
# - Disruptive and creative solutions
# - Global cultural sensitivity
# - Sustainability and social responsibility
# - Customer-centric innovation
# - Long-term brand and societal impact

# Team Argument: {input}

# Deliver a transformative and visionary assessment."""
# }

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get('question', '')
    level = data.get('level', 1)

    if not user_question:
        return jsonify({"error": "No question provided."}), 400

    # Initialize Groq LLM
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        #model_name='mixtral-8x7b-32768'
        model_name='llama3-8b-8192'

    )

    # Create a prompt template for the specific leadership level
    prompt_template = PromptTemplate(
        input_variables=["input"],
        template=LEVEL_PROMPTS.get(level, LEVEL_PROMPTS[1])
    )

    # Create an LLM Chain
    chain = LLMChain(llm=groq_chat, prompt=prompt_template)

    # Generate response
    response = chain.run(input=user_question)

    return jsonify({"response": response})

@app.route('/clear', methods=['POST'])
def clear_history():
    # Simple clear route (you can expand this if needed)
    return jsonify({"message": "Session reset."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
