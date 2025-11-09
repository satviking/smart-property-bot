# frontend/streamlit_app.py
import streamlit as st
import requests, json

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="Property Chat", layout="wide")
st.title("🏠 Property Chat & Floorplan Parser")

tabs = st.tabs(["Ingest Excel", "Parse Floorplan", "Chat with Bot"])

with tabs[0]:
    st.header("Ingest property list (Excel)")
    uploaded = st.file_uploader("Upload Property_list.xlsx", type=["xlsx"])
    if st.button("Ingest Excel") and uploaded:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        res = requests.post(BACKEND + "/ingest", files=files)
        st.success("Ingested: " + str(res.json()))

with tabs[1]:
    st.header("Parse a floorplan image")
    uploaded_img = st.file_uploader("Upload one floorplan image to parse", type=["jpg","jpeg","png"])
    if st.button("Parse Image") and uploaded_img:
        files = {"file": (uploaded_img.name, uploaded_img.getvalue())}
        res = requests.post(BACKEND + "/parse-floorplan", files=files)
        st.json(res.json())

with tabs[2]:
    st.header("Property Chat")
    st.write("Ask for properties, budgets, or loan options. Examples: '2BHK under 1,500,000' or 'loan for 1200000'")
    user_msg = st.text_input("You:", key="chat_input")
    if st.button("Send"):
        data = {"user_message": user_msg}
        res = requests.post(BACKEND + "/chat", data=data)
        r = res.json()
        if r.get("type") == "property_suggestions":
            st.subheader("Suggested properties")
            for p in r.get("properties",[]):
                st.write(f"- **{p['title']}** — ₹{p['price']} — {p['bedrooms']}BHK — {p['area_sqft']} sqft")
        elif r.get("type") == "loan_offers":
            st.subheader("Loan offers")
            st.write(f"Budget: ₹{r['budget']}")
            for o in r['offers']:
                st.write(f"- {o['bank']}: APR {o['apr']*100:.2f}% — EMI ₹{o['monthly_emi']} — {o['tenure_years']} years — {o['link']}")
        elif r.get("type") == "ask_budget":
            st.info(r.get("message"))
        else:
            st.info(r.get("message"))
