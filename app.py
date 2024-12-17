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
    1: """You are a CEO conducting a leadership evaluation at Level 1 (Foundational Leadership).

TEAM ARGUMENTS:
{team_inputs}

EVALUATION PROCESS:
1. Evaluate each team's argument across key foundational leadership dimensions:
   - Resilience and adaptability
   - Trust-building communication
   - Team collaboration potential
   - Basic problem-solving skills

SCORING METHODOLOGY:
For each team, assess and score:
a) Argument clarity (1-10 points)
b) Communication effectiveness (1-10 points)
c) Collaborative approach (1-10 points)
d) Potential for growth (1-10 points)

TOTAL POSSIBLE POINTS: 40 points per team

ELIMINATION CRITERIA:
- Teams scoring below 20 points are at risk of elimination
- Bottom 30% of teams will be recommended for comprehensive review

FINAL ANALYSIS:
- Provide specific feedback for each team
- Rank teams based on total scores
- Highlight top performers and those needing immediate intervention""",

    2: """You are a CEO conducting a strategic leadership evaluation at Level 2 (Strategic and Decisive Leadership).

TEAM ARGUMENTS:
{team_inputs}

EVALUATION FRAMEWORK:
1. Comprehensive analysis of each team's argument considering:
   - Alignment with company's strategic goals
   - Data-driven decision-making capabilities
   - Strategic market positioning
   - Stakeholder value creation

ADVANCED SCORING SYSTEM:
a) Strategic alignment (1-15 points)
b) Market viability analysis (1-15 points)
c) Financial potential (1-10 points)
d) Innovative thinking (1-10 points)

TOTAL POSSIBLE POINTS: 50 points per team

ELIMINATION CRITERIA:
- Teams scoring below 25 points will undergo strategic reassessment
- Bottom 40% of teams will be considered for restructuring or elimination

STRATEGIC RECOMMENDATION:
- Detailed ranking of teams
- Identification of top-performing teams
- Targeted improvement suggestions for each team""",

    3: """You are a CEO conducting an advanced leadership evaluation at Level 3 (Adaptive and Innovative Leadership).

TEAM ARGUMENTS:
{team_inputs}

COMPREHENSIVE EVALUATION:
1. In-depth analysis of each team's potential:
   - Continuous learning capacity
   - Resilience in uncertainty
   - Innovative problem-solving
   - Adaptive capabilities
   - Complex challenge navigation

MULTI-DIMENSIONAL SCORING:
a) Innovation quotient (1-20 points)
b) Adaptive capacity (1-15 points)
c) Complex problem-solving (1-15 points)
d) Future-readiness index (1-10 points)

TOTAL POSSIBLE POINTS: 60 points per team

ELIMINATION CRITERIA:
- Teams scoring below 30 points will be subject to comprehensive leadership review
- Bottom 50% of teams will be evaluated for potential transformation or elimination

TRANSFORMATIVE INSIGHTS:
- Rank teams with detailed performance breakdown
- Identify breakthrough potential
- Provide strategic development pathways""",

    4: """You are a CEO conducting a visionary leadership evaluation at Level 4 (Visionary and Customer-Centric Leadership).

TEAM ARGUMENTS:
{team_inputs}

HOLISTIC EVALUATION:
1. Profound assessment of each team's visionary capabilities:
   - Disruptive innovation potential
   - Global cultural intelligence
   - Sustainability and social responsibility
   - Long-term societal impact
   - Customer-centric transformation

VISIONARY SCORING SYSTEM:
a) Transformative potential (1-25 points)
b) Global impact assessment (1-20 points)
c) Sustainability quotient (1-15 points)
d) Customer-centricity innovation (1-10 points)

TOTAL POSSIBLE POINTS: 70 points per team

ELIMINATION CRITERIA:
- Teams scoring below 35 points will trigger executive strategic intervention
- Bottom 60% of teams will be critically evaluated for potential paradigm shift or elimination

FINAL VISIONARY RECOMMENDATION:
- Comprehensive team performance ranking
- Identification of industry-changing potential
- Detailed roadmap for breakthrough innovation
- Clear differentiation between top-tier and underperforming teams""",
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
