import os
import json
import pandas as pd
import traceback
# import PyPDF2
# from pathlib import Path
# from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv() # Load environment variables from the .env file

from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging 

llm = ChatOpenAI(
    api_key=os.getenv('GROQ_API_KEY'), # Access the environment variables 
    base_url="https://api.groq.com/openai/v1",
    model="openai/gpt-oss-120b",
    temperature=0.5,
)

template = """
Text: {text}
You are an expert NCQ maker. Given the above text, it is your job to create 
a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. 
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template = template
)

quiz_chain = quiz_generation_prompt | llm | StrOutputParser()

template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject}
students. You need to evaluate the complexity of the question and give a complete analysis of
the quiz. Only use at max 50 words for complexity if the quiz is not at per with the cognitive 
and analytical abilities of the students. Update the quiz questions which needs to be changed
and change the tone such that it perfectly fits the students' ability.
Quiz MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables= ["subject", "quiz"],
    template = template2
)

review_chain = quiz_evaluation_prompt | llm | StrOutputParser()

generate_evaluate_chain = (
    RunnablePassthrough()
    .assign(quiz=quiz_chain)
    .assign(review=review_chain)
)