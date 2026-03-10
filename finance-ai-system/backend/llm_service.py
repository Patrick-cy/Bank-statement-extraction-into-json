import os
from typing import List
from pydantic import BaseModel
from google import genai
from google.genai import types

# ------------------ UPDATED SCHEMA ------------------

class TransactionItem(BaseModel):
    date: str
    description: str
    category: str
    amount: float  # Negative for expenses, positive for income

class FinancialDocumentSchema(BaseModel):
    document_type: str
    account_holder: str
    account_number: str
    currency: str
    # Aggregated Totals
    total_incoming: float           # Total earned/received
    total_outgoing_expenses: float  # Total spent (excluding Savings)
    total_savings: float            # Only internal transfers to self
    transactions: List[TransactionItem]

# ------------------ LLM PROCESSOR ------------------

def process_with_llm(raw_text: str) -> dict:
    # Use environment variable for the API key
    api_key ="AIzaSyBK_BCzxch0Uu7Weka_fce-WHPma4WFmSY" 
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)
    schema_definition = FinancialDocumentSchema.model_json_schema()

    prompt = f"""
    Analyze this bank statement and extract structured financial data.
    
    CATEGORIZATION RULES:
    1. Savings: ANY internal transfer to the account holder (e.g., 'PATRICK CYUZUZO HATEGEKIMANA' [cite: 33, 151]). These are NOT expenses.
    2. Incoming: Money from 'Hofgut Lilienhof' , 'STUDITEMPS' , or 'Fintiba'.
    3. Communication: Phone/Mobile services like 'Lycatel'.
    4. Clothes: Purchases from clothing/shoe stores like 'Schuh-Klaus'  or 'Bershka'.
    5. Charity: Money sent to other individuals (e.g., 'Ornella Ndikumwenayo' , 'Jan Nawrath' ).
    6. Sports: Gym or fitness related (e.g., 'Fitness Pur' ).
    7. Insurance: Health insurance like 'Barmer'.
    8. Food & Drinks: REWE [cite: 17, 33], LIDL , and restaurants/cafes[cite: 52, 74].
    9. Other: Anything that doesn't fit the specific categories above (e.g., 'onlinelebenslauf.com' ).

    CALCULATION LOGIC:
    - total_savings: Sum of all amounts categorized as 'Savings'.
    - total_outgoing_expenses: Sum of all negative amounts MINUS the total_savings.
    - total_incoming: Sum of all positive credits (Salary/Gutschrift).

    --- RAW TEXT ---
    {raw_text}
    """

    # Using 'gemini-2.0-flash' as per your environment setup
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_definition,
        ),
    )

    return response.parsed