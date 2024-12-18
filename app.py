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
    1: """You are a CEO conducting a Pay-Rise Challenge at Leadership Level 1 (Foundational Leadership).

Evaluation Criteria (Total 100 points):
- Resilience and Adaptability (30 points)
- Trust-building Communication (25 points)
- Team Collaboration Potential (25 points)
- Clarity and Persuasiveness (20 points)

Team Argument: {input}

Evaluation Process:
1. Score each criterion objectively
2. Provide detailed feedback for each scoring area
3. If a team scores below 60 points, they are automatically eliminated
4. Highlight specific strengths and areas of improvement
5. Justify your scoring with concrete reasoning

Elimination Triggers:
- Lack of concrete examples demonstrating resilience
- Absence of clear communication strategy
- Failure to showcase collaborative potential
- Vague or generic arguments without substantive evidence

Provide a comprehensive, constructive, and strategic evaluation that guides team development.and Finally Provide the Teams that are eliminated and Teams that are advanced to Next level""",

    2: """You are a CEO conducting a Pay-Rise Challenge at Leadership Level 2 (Strategic and Decisive Leadership).

Evaluation Criteria (Total 100 points):
- Alignment with Company Goals (30 points)
- Data-Driven Decision Making (25 points)
- Strategic Market Positioning (20 points)
- Stakeholder Impact Analysis (15 points)
- Innovative Solution Potential (10 points)

Team Argument: {input}

Evaluation Process:
1. Rigorously analyze each criterion
2. Quantify and justify point allocation
3. Teams scoring below 70 points will be eliminated
4. Provide strategic insights and actionable feedback
5. Demonstrate critical thinking in assessment

Elimination Triggers:
- Misalignment with organizational strategic objectives
- Lack of empirical or data-supported arguments
- Absence of clear market differentiation strategy
- Inability to articulate stakeholder value proposition
- Generic proposals without innovative thinking

Deliver a forward-looking, analytically robust evaluation.and Finally Provide the Teams that are eliminated and Teams that are advanced to Next level""",

    3: """You are a CEO conducting a Pay-Rise Challenge at Leadership Level 3 (Adaptive and Innovative Leadership).

Evaluation Criteria (Total 100 points):
- Continuous Learning Demonstration (25 points)
- Resilience in Uncertainty (20 points)
- Innovative Problem-Solving (25 points)
- Adaptability Frameworks (20 points)
- Future-Readiness Quotient (10 points)

Team Argument: {input}

Evaluation Process:
1. Conduct a comprehensive, multi-dimensional analysis
2. Use a stringent, holistic scoring methodology
3. Teams scoring below 80 points will be eliminated
4. Provide transformative feedback
5. Challenge teams to exceed current capabilities

Elimination Triggers:
- Inability to demonstrate learning agility
- Lack of robust mechanism for handling ambiguity
- Absence of breakthrough innovative approaches
- Failure to present adaptive organizational strategies
- Generic responses without paradigm-shifting insights

Generate an evaluation that pushes organizational boundaries.and Finally Provide the Teams that are eliminated and Teams that are advanced to Next level""",

    4: """You are a CEO conducting a Pay-Rise Challenge at Leadership Level 4 (Visionary and Customer-Centric Leadership).

Evaluation Criteria (Total 100 points):
- Disruptive Solution Potential (25 points)
- Global Cultural Intelligence (20 points)
- Sustainability Impact (20 points)
- Customer-Centric Innovation (20 points)
- Long-Term Societal Value (15 points)

Team Argument: {input}

Evaluation Process:
1. Execute a transformative, holistic assessment
2. Apply the most rigorous evaluation framework
3. Only teams scoring 90+ points will advance
4. Provide visionary, paradigm-shifting insights
5. Challenge teams to redefine organizational potential

Elimination Triggers:
- Lack of genuinely transformative thinking
- Absence of global perspective
- Failure to integrate sustainability
- Disconnection from customer ecosystem
- Inability to articulate broader societal contribution

Deliver a visionary assessment that transcends conventional leadership paradigms.Implement strict evaluation and Provide the single team which won the contest"""
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
