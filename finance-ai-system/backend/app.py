import os
import pdfplumber
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from config import Config
from models import db, Document, Transaction
from llm_service import process_with_llm

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

app.config.from_object(Config)
db.init_app(app)

# ---------------- INIT DB ----------------
with app.app_context():
    db.create_all()

# ---------------- HELPER FUNCTION ----------------
def parse_date(date_string):
    """
    Safely parse date strings from AI output.
    """
    if not date_string:
        return None
    
    # Remove any potential timestamps or extra whitespace
    date_clean = date_string.split('T')[0].strip()
    
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_clean, fmt).date()
        except ValueError:
            continue
    return None

# ---------------- UPLOAD + AI ----------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        os.makedirs("uploads", exist_ok=True)
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Extract text from PDF
        text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        
        if not text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # AI Processing - Using updated llm_service
        json_data = process_with_llm(text)

        # Validate required keys based on updated schema
        required_keys = [
            "document_type", "vendor_name", "document_date", 
            "currency", "total_incoming", "total_outgoing_expenses", 
            "total_savings", "transactions"
        ]
        for key in required_keys:
            if key not in json_data:
                return jsonify({"error": f"Missing key in AI response: {key}"}), 500

        # Save Document with new summary fields
        doc = Document(
            document_type=json_data["document_type"],
            vendor_name=json_data["vendor_name"],
            document_date=parse_date(json_data["document_date"]),
            currency=json_data["currency"],
            total_incoming=float(json_data["total_incoming"]),
            total_expenses=float(json_data["total_outgoing_expenses"]),
            total_savings=float(json_data["total_savings"])
        )
        db.session.add(doc)
        db.session.flush() # Flushes to DB to generate doc.id

        # Save Transactions
        for tx in json_data["transactions"]:
            # Note: Category is now strictly enforced by the prompt instructions
            transaction = Transaction(
                document_id=doc.id,
                description=tx["description"],
                amount=float(tx["amount"]), # Changed from line_total to amount
                transaction_date=parse_date(tx["date"]),
                category=tx["category"]
            )
            db.session.add(transaction)

        db.session.commit()

        return jsonify({
            "message": "Statement processed successfully",
            "document_id": doc.id,
            "summary": {
                "incoming": json_data["total_incoming"],
                "expenses": json_data["total_outgoing_expenses"],
                "savings": json_data["total_savings"]
            }
        }), 200

    except Exception as e:
        traceback.print_exc()
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# ---------------- ANALYTICS ----------------
@app.route("/analytics/category")
def category_summary():
    # Sums all transactions by category
    results = db.session.query(
        Transaction.category,
        db.func.sum(Transaction.amount)
    ).group_by(Transaction.category).all()

    return jsonify([
        {"category": r[0], "total": float(r[1])}
        for r in results
    ])

if __name__ == "__main__":
    app.run(debug=True, port=5000)