import os
import pdfplumber
import json
import time 
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from google import genai
from google.genai import types
from google.genai import errors as genai_exceptions 


os.environ['GEMINI_API_KEY'] = 'AIzaSyB9XEAxYBzSH2oeNCbhE0zvazUzlxvLOb8'
print("GEMINI_API_KEY set successfully.")


PDF_PATH = "/Users/patrickcyuzuzo/Downloads/Statement (1).pdf" 



class TransactionItem(BaseModel):
    """Schema for a single line item or transaction detail."""
    description: str = Field(description="Description of the item, service, or transaction.")
    quantity_or_amount: float = Field(description="The quantity (for receipts) or the value/amount (for statements/receipts). Use 0.0 if not applicable.")
    unit_price: float = Field(description="The price per unit, if available. Use 0.0 if not applicable.")
    line_total: float = Field(description="The total monetary value for this specific item or transaction.")
    date: str = Field(description="Date of the transaction/item in YYYY-MM-DD format (use 'N/A' if only a date range is given).")

class FinancialDocumentSchema(BaseModel):
    """Schema for extracting data from a Receipt or Bank Statement."""
    document_type: str = Field(description="The type of document, MUST be 'Receipt' or 'Bank Statement'.")
    vendor_name: str = Field(description="The name of the seller, business, or bank.")
    document_date: str = Field(description="The primary date of the document (issue date, period end date) in YYYY-MM-DD format.")
    currency: str = Field(description="The primary currency used (e.g., 'USD', 'EUR').")
    
    subtotal: float = Field(description="The total cost before tax, tips, or fees. Use 0.0 if not found.")
    tax_amount: float = Field(description="The total tax amount paid. Use 0.0 if not found.")
    total_amount_due: float = Field(description="The final total monetary value to be paid or the statement's closing balance.")
    
    account_number: str = Field(description="The account number, if available. Use 'N/A' if not found.")
    statement_period: str = Field(description="The period covered (e.g., '2024-05-01 to 2024-05-31'). Use 'N/A' if not applicable.")

    transactions: List[TransactionItem] = Field(description="A list of all individual line items (for a receipt) or transactions (for a statement).")



def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Extracts raw text content from all pages of a PDF."""
    print(f"-> Extracting text from {pdf_file_path}...")
    try:
        all_text = ""
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() + "\n--- PAGE BREAK ---\n"
        return all_text
    except FileNotFoundError:
        print(f"ERROR: PDF file not found at {pdf_file_path}")
        return ""


def convert_to_json_with_llm(raw_text: str, client: genai.Client) -> dict:
    """Uses the LLM to extract financial data based on the specialized schema."""
    
    # 1. Use the new FinancialDocumentSchema
    schema_definition = FinancialDocumentSchema.model_json_schema()

    print("-> Sending text to LLM for Financial Extraction (using gemini-2.5-flash)...")
    
    # 2. Specialized Prompt for Financial Docs (REPLACING GENERIC PROMPT)
    prompt = f"""
    Analyze the following raw text. This document must be treated as either a **Receipt** or a **Bank Statement**.
    Your task is to parse all financial and transactional details and strictly populate the JSON fields according to the provided schema (FinancialDocumentSchema).
    
    - Determine if it is a 'Receipt' or 'Bank Statement' for the document_type field.
    - Accurately convert all monetary values (subtotal, tax, total, transaction totals) to floating point numbers (e.g., 15.50). Use 0.0 if a field is not present.
    - Extract all individual line items or bank transactions and populate the 'transactions' list, ensuring the 'date' and 'line_total' are extracted for each.

    --- RAW TEXT ---
    {raw_text}
    """
    
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            # Configure the Model for Structured Output (JSON Mode)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_definition,
                )
            )
            
            # Successful response, proceed to parsing
            json_output = response.text
            parsed_data = FinancialDocumentSchema.parse_raw(json_output)
            return parsed_data.dict()

        # Error Handling (Same as before)
        except genai_exceptions.ClientError as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed (Client Error, likely 429 Quota).")
            if attempt < MAX_RETRIES - 1:
                delay = 5 * (2 ** attempt) 
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("Max retries reached for ClientError. Failing conversion.")
                return {}
        
        except genai_exceptions.ServerError as e:
            print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed (Server Error, likely 503 Unavailable).")
            if attempt < MAX_RETRIES - 1:
                delay = 3 * (2 ** attempt) 
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("Max retries reached for ServerError. Failing conversion.")
                return {}
                
        except Exception as e:
            print(f"An unexpected error occurred during API call or Pydantic parsing: {e}")
            return {}
            


def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("FATAL ERROR: The GEMINI_API_KEY environment variable is not set.")
        return

    try:
        client = genai.Client() 
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}. Ensure API key setup is complete.")
        return

  
    raw_text = extract_text_from_pdf(PDF_PATH)
    if not raw_text:
        return


    json_data = convert_to_json_with_llm(raw_text, client)

 
    if json_data:

        output_filename = PDF_PATH.replace('.pdf', '_financial_extract.json') 
        
      
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        
        print("\n✅ Success!")
        print(f"Financial JSON saved to: {output_filename}")
        print("-" * 50)
        
        print(f"Document Type: **{json_data.get('document_type', 'N/A')}**")
        print(f"Vendor/Bank: {json_data.get('vendor_name', 'N/A')}")
        print(f"Total Amount: {json_data.get('total_amount_due', 'N/A')} {json_data.get('currency', 'N/A')}")
        print(f"Transactions Extracted: {len(json_data.get('transactions', []))}")
        
        print("\n--- Extracted JSON Preview ---")
        # Print the extracted dictionary in a clean, readable JSON format
        print(json.dumps(json_data, indent=4))
        print("-" * 50)
        
    else:
        print("\n Failed to generate structured JSON output.")

if __name__ == "__main__":
    main()