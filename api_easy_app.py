import os
import io
import pdfplumber
import json
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
CORS(app)

db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'Hategekimana',
    'database': 'easy_app'
}

# --- SCHEMAS ---
class TransactionItem(BaseModel):
    description: str = Field(description="Description of the item.")
    line_total: float = Field(description="Total for this specific line.")
    date: str = Field(description="YYYY-MM-DD")

class FinancialDocumentSchema(BaseModel):
    document_type: str = Field(description="'Receipt' or 'Bank Statement'")
    vendor_name: str = Field(description="Business/Bank name")
    document_date: str = Field(description="YYYY-MM-DD")
    currency: str = Field(default="USD")
    total_amount_due: float = Field(description="Final total")
    transactions: List[TransactionItem]

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf_binary(pdf_blob):
    """Reads PDF text from a binary stream instead of a file path."""
    all_text = ""
    # Use io.BytesIO to treat binary data like a file
    with pdfplumber.open(io.BytesIO(pdf_blob)) as pdf:
        for page in pdf.pages:
            all_text += (page.extract_text() or "") + "\n"
    return all_text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def convert_to_json_with_llm(raw_text, client):
    schema_definition = FinancialDocumentSchema.model_json_schema()
    prompt = f"Analyze this text as a Receipt or Bank Statement:\n{raw_text}"
    
    response = client.models.generate_content(
        model='gemini-1.5-flash', 
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_definition,
        )
    )
    return json.loads(response.text)

# --- API ROUTES ---

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    filename = file.filename
    file_binary = file.read() # Read the actual file data

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 1. SAVE FILE TO DATABASE FIRST
        # Note: Ensure your 'documents' table has a 'file_data' LONGBLOB column
        insert_doc_query = """
            INSERT INTO documents (filename, file_data) 
            VALUES (%s, %s)
        """
        cursor.execute(insert_doc_query, (filename, file_binary))
        doc_id = cursor.lastrowid
        conn.commit()

        # 2. READ FILE BACK FROM DATABASE
        cursor.execute("SELECT file_data FROM documents WHERE id = %s", (doc_id,))
        record = cursor.fetchone()
        db_blob = record['file_data']

        # 3. TRANSFORM TO JSON USING LLM
        raw_text = extract_text_from_pdf_binary(db_blob)
        client = genai.Client(api_key='AIzaSyDAalNYe2-lxUCWuTZWVvn6pKMzT9h8z94') 
        json_data = convert_to_json_with_llm(raw_text, client)

        # 4. UPDATE RECORD WITH EXTRACTED DATA
        update_query = """
            UPDATE documents 
            SET vendor_name = %s, total_amount = %s, currency = %s 
            WHERE id = %s
        """
        cursor.execute(update_query, (
            json_data['vendor_name'], 
            json_data['total_amount_due'], 
            json_data['currency'], 
            doc_id
        ))

        for tx in json_data['transactions']:
            cursor.execute(
                "INSERT INTO transactions (document_id, date, description, amount) VALUES (%s, %s, %s, %s)",
                (doc_id, tx['date'], tx['description'], tx['line_total'])
            )
        
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Success", "doc_id": doc_id, "data": json_data})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)