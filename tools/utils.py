
import json
import os
import pandas as pd
import random
from openai import OpenAI
import google.generativeai as genai

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
client = OpenAI()
genai.configure(api_key='GEMINI_API_KEY')

def gpt_response(prompt, gpt_version='gpt-4.1-mini'):
    try:
        response = client.chat.completions.create(
            model=gpt_version,
            messages=[{"role": "user", "content": prompt}],
            timeout=20
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred: {e}. Retrying...")

        return 'None'

import time
def gemini_response(prompt, model_name="gemini-1.5-pro"): # gemini-1.5-flash gemini-1.5-pro
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    time.sleep(0.1)  # Sleep for 0.1 second
    return response.text
