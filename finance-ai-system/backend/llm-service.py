import os
from typing import List
from pydantic import BaseModel
from google import genai
from google.genai import types


# ------------------ SCHEMA ------------------

class TransactionItem(BaseModel):
    description: str
    quantity_or_amount: float
    unit_price: float
    line_total: float
    date: str
    category: str


class FinancialDocumentSchema(BaseModel):
    document_type: str
    vendor_name: str
    document_date: str
    currency: str
    subtotal: float
    tax_amount: float
    total_amount_due: float
    account_number: str
    statement_period: str
    transactions: List[TransactionItem]


# ------------------ LLM PROCESSOR ------------------

def process_with_llm(raw_text: str) -> dict:

    client = genai.Client(api_key=os.getenv("AIzaSyB9XEAxYBzSH2oeNCbhE0zvazUzlxvLOb8"))

    schema_definition = FinancialDocumentSchema.model_json_schema()

    prompt = f"""
    Analyze this financial document (bank statement or receipt).

    Extract structured financial data.

    For EACH transaction classify into one of:
    Education
    Food & Drinks
    Utilities
    Phone
    Tax
    Clothes
    Travel 
    Leisure
    Insurance
    Rent
    Sports
    Electricity
    Wifi
    Charity
    Loans
    Other

    Rules:
    - Money sent to a person → Charity
    -Money paid for hotels, airbnb, booking.com → Leisure
    - Loan repayment → Loans
    - Money paid for Gym  → Sports
    - Electricity bill → Electricity
    - Internet bill → Wifi

    --- RAW TEXT ---
    {raw_text}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_definition,
        ),
    )

    parsed = FinancialDocumentSchema.parse_raw(response.text)
    return parsed.dict()