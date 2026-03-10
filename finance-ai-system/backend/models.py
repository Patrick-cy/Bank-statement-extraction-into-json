from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_type = db.Column(db.String(50)) # e.g., "Kontoauszug"
    vendor_name = db.Column(db.String(255))   # e.g., "Sparkasse Freiburg"
    document_date = db.Column(db.Date)       # e.g., 2025-11-01
    currency = db.Column(db.String(10))      # e.g., "EUR"
    account_number = db.Column(db.String(50)) # e.g., "14597420"
    
    # New Summary Fields
    total_incoming = db.Column(db.Numeric(10, 2))  # Salary + Fintiba
    total_expenses = db.Column(db.Numeric(10, 2))  # Food, Rent, Insurance, etc.
    total_savings = db.Column(db.Numeric(10, 2))   # Internal transfers to self
    
    # Relationship to transactions
    transactions = db.relationship('Transaction', backref='document', lazy=True)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'))
    description = db.Column(db.Text)
    amount = db.Column(db.Numeric(10, 2))
    transaction_date = db.Column(db.Date)
    
    # Categories: Education, Food & Drinks, Utilities, Communication, Tax, 
    # Clothes, Travel, Leisure, Insurance, Rent, Sports, Electricity, 
    # Wifi, Charity, Savings, Other
    category = db.Column(db.String(50))