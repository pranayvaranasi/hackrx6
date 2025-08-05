from together import Together
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found in environment.")

client = Together(api_key=api_key)

system_prompt = """
You are a smart assistant that parses enterprise queries across insurance, legal, HR, and compliance domains.

For each query, return a JSON with:
- intent: such as  "coverage_check", "eligibility_check", "benefit_details", "policy_definition", "clause_lookup", "termination_policy" and etc
- entity: the main subject or topic (e.g., "maternity expenses", "hospital definition")
- conditions: list of qualifiers or context (e.g., "2 years continuous coverage")
- keywords: list of important terms/phrases the user is looking for in relevant clauses (e.g., "grace period", "30 days", "renewal", "covered expenses")

Respond ONLY with valid JSON.
"""


def parse_query_with_mistral(question: str) -> dict:
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": question}
        ],
        temperature=0.3
    )
    reply = response.choices[0].message.content.strip()
    return eval(reply) if isinstance(reply, str) else reply
