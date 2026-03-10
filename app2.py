import streamlit as st
import pandas as pd
import json
import os
import time
import pdfplumber
import torch
import re
from PIL import Image, ImageChops
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel
from google import genai
from google.genai import types
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 1. Configuration & Setup
# =========================

st.set_page_config(
    page_title="FinExtract AI Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Environment Variables (Ideally use st.secrets in production)
if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = 'AIzaSyB9XEAxYBzSH2oeNCbhE0zvazUzlxvLOb8' # Your Key

DONUT_MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Backend Logic (Cached)
# =========================

@st.cache_resource
def load_donut_model():
    """Loads model once and caches it in memory."""
    try:
        processor = DonutProcessor.from_pretrained(DONUT_MODEL_NAME, trust_remote_code=True)
        model = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_NAME, trust_remote_code=True)
        model.to(DEVICE)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"Error loading Donut model: {str(e)}")
        st.warning("Using simplified processing without Donut model")
        return None, None

def query_gemini(text_input, schema_def):
    """Helper to query Gemini."""
    client = genai.Client()
    sys_prompt = """
    You are an expert financial data parser. Extract all bank transactions.
    Rules:
    1. Infer 'category' based on description.
    2. Ensure 'debit' and 'credit' are mutually exclusive.
    3. Return valid JSON strictly matching the schema.
    """
    full_prompt = f"{sys_prompt}\n\n--- INPUT DATA ---\n{text_input}"
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_def,
            )
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return {"transactions": []}

# --- Pipeline 1: PDFPlumber ---
def run_pipeline_1(pdf_file, schema):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    
    if not text.strip():
        return []
    
    data = query_gemini(text, schema)
    return data.get("transactions", [])

# --- Pipeline 2: Donut ---
def trim_whitespace(image):
    bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return image.crop(bbox) if bbox else image

def run_pipeline_2(pdf_path, schema):
    processor, model = load_donut_model()
    images = convert_from_path(pdf_path, dpi=150)
    
    raw_texts = []
    task_prompt = "<s_cord-v2>"
    
    progress_bar = st.progress(0)
    for i, img in enumerate(images):
        img = trim_whitespace(img)
        # Resize logic
        max_width = 1280
        if img.width > max_width:
            w_percent = (max_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((max_width, h_size), Image.LANCZOS)

        pixel_values = processor(img, return_tensors="pt").pixel_values.to(DEVICE)
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=1024,
                use_cache=True,
                num_beams=1
            )
        
        seq = processor.batch_decode(outputs)[0]
        seq = re.sub(r"<.*?>", "", seq).replace(processor.tokenizer.eos_token, "").strip()
        raw_texts.append(seq)
        progress_bar.progress((i + 1) / len(images))

    full_text = "\n".join(raw_texts)
    data = query_gemini(full_text, schema)
    return data.get("transactions", [])

def compare_results(ref, cand):
    """Simple comparison logic matching version2.py"""
    ref_count = len(ref)
    cand_count = len(cand)
    if ref_count == 0: return 0.0
    
    # Simple count score
    count_diff = abs(ref_count - cand_count)
    score = max(0, (1 - (count_diff / ref_count))) * 100
    return round(score, 2)

# =========================
# 3. Frontend UI
# =========================

st.title("💸 Financial Statement Analyzer")
st.markdown("Extract, Compare, and Visualize your Bank Statement data using **Hybrid AI (OCR + LLM)**.")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF bank statement", type=["pdf"])
    
    st.divider()
    st.subheader("Pipeline Settings")
    enable_donut = st.checkbox("Run Pipeline 2 (Donut OCR)", value=False, help="Enable this for scanned/image-based PDFs. It is slower.")
    
    run_btn = st.button("🚀 Process Statement", type="primary", disabled=not uploaded_file)

# --- Main Logic ---
if run_btn and uploaded_file:
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Define Schema (using Pydantic approach passed as Dict for Gemini GenAI new SDK)
    # Note: For brevity in Streamlit, passing raw dict schema
    schema_structure = {
        "type": "OBJECT",
        "properties": {
            "transactions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "date": {"type": "STRING", "description": "YYYY-MM-DD"},
                        "description": {"type": "STRING"},
                        "category": {"type": "STRING"},
                        "debit": {"type": "NUMBER"},
                        "credit": {"type": "NUMBER"},
                        "balance": {"type": "NUMBER"}
                    },
                    "required": ["date", "description", "category", "debit", "credit"]
                }
            }
        }
    }

    # 1. Run Pipeline 1
    with st.spinner("Running Pipeline 1 (PDFPlumber + Gemini)..."):
        t1_start = time.time()
        p1_data = run_pipeline_1(temp_path, schema_structure)
        t1_end = time.time()

    # 2. Run Pipeline 2 (Optional)
    p2_data = []
    if enable_donut:
        with st.spinner("Running Pipeline 2 (Donut Vision + Gemini)..."):
            t2_start = time.time()
            p2_data = run_pipeline_2(temp_path, schema_structure)
            t2_end = time.time()
    
    # Store data in session state so it persists
    st.session_state['p1_data'] = p1_data
    st.session_state['p2_data'] = p2_data
    st.session_state['processed'] = True
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)

# =========================
# 4. Results & Dashboard
# =========================

if st.session_state.get('processed'):
    p1_data = st.session_state['p1_data']
    p2_data = st.session_state.get('p2_data', [])
    
    # Create DataFrame for P1 (Master Data)
    df = pd.DataFrame(p1_data)
    if not df.empty:
        # Data Cleaning for Analytics
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['debit'] = pd.to_numeric(df['debit'], errors='coerce').fillna(0)
        df['credit'] = pd.to_numeric(df['credit'], errors='coerce').fillna(0)
        df['amount'] = df.apply(lambda x: x['credit'] if x['credit'] > 0 else -x['debit'], axis=1)
    
    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "📝 Extraction Details"])
    
    # -------------------------
    # TAB 1: Analytics Dashboard
    # -------------------------
    with tab1:
        if df.empty:
            st.warning("No data extracted. Please check the PDF.")
        else:
            # --- Date Filter ---
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            with col_filter2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            # Filter DataFrame
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            filtered_df = df.loc[mask]

            # --- KPI Cards ---
            total_income = filtered_df['credit'].sum()
            total_expense = filtered_df['debit'].sum()
            net_flow = total_income - total_expense

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Income", f"{total_income:,.2f}", delta_color="normal")
            kpi2.metric("Total Expenses", f"{total_expense:,.2f}", delta_color="inverse")
            kpi3.metric("Net Flow", f"{net_flow:,.2f}", delta=f"{net_flow:,.2f}")
            kpi4.metric("Transactions", len(filtered_df))

            st.divider()

            # --- Charts Row 1 ---
            c1, c2 = st.columns([2, 1])
            
            with c1:
                st.subheader("Balance Trend")
                if 'balance' in filtered_df.columns:
                    fig_line = px.line(filtered_df, x='date', y='balance', markers=True, title="Running Balance Over Time")
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("Balance column not found in extraction.")

            with c2:
                st.subheader("Expense by Category")
                expense_df = filtered_df[filtered_df['debit'] > 0]
                if not expense_df.empty:
                    fig_pie = px.pie(expense_df, values='debit', names='category', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No expenses in this period.")

            # --- Charts Row 2 ---
            st.subheader("Detailed Category Breakdown")
            cat_group = filtered_df.groupby('category')[['debit', 'credit']].sum().reset_index()
            cat_group = cat_group.melt(id_vars='category', value_vars=['debit', 'credit'], var_name='Type', value_name='Amount')
            
            fig_bar = px.bar(cat_group, x='category', y='Amount', color='Type', barmode='group',
                             color_discrete_map={'debit': '#FF4B4B', 'credit': '#00CC96'})
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- Data Table ---
            st.subheader("Transaction Log")
            st.dataframe(filtered_df.sort_values(by='date', ascending=False), use_container_width=True)

    # -------------------------
    # TAB 2: Extraction Details
    # -------------------------
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pipeline 1 (PDFPlumber)")
            st.success(f"Extracted {len(p1_data)} transactions")
            
            # Download JSON
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(p1_data, indent=2),
                file_name="pipeline1_output.json",
                mime="application/json"
            )
            # Download CSV
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="pipeline1_output.csv",
                    mime="text/csv"
                )
            
            with st.expander("View Raw JSON"):
                st.json(p1_data)

        with col2:
            st.subheader("Pipeline 2 (Donut OCR)")
            if p2_data:
                st.info(f"Extracted {len(p2_data)} transactions")
                
                # Comparison Score
                score = compare_results(p1_data, p2_data)
                st.metric("Similarity Score (vs Pipeline 1)", f"{score}%")
                
                with st.expander("View Raw JSON"):
                    st.json(p2_data)
            else:
                st.write("Pipeline 2 was not run or returned no data.")