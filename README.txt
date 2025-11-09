🏠 Smart Property Bot Backend

Automated Floorplan Understanding and Property Intelligence System

This project integrates YOLOv8 (custom-trained) and OCR (Tesseract + EasyOCR) into a FastAPI backend that can:

Parse architectural floorplans

Extract dimensions and room details

Infer bedrooms, bathrooms, and total area

Manage property data via database + Excel ingestion

Provide intelligent chat-based property search and loan suggestions

Core Features
Feature	Description
YOLOv8 Model Integration	Detects floorplan elements like rooms, dimensions, ceilings, etc. using your trained model best_scratch.pt (400 epochs).
OCR Extraction (Tesseract + EasyOCR)	Reads room labels and dimension text from images, automatically calculating area in sq. ft.
Excel Data Ingestion	Upload an Excel sheet of properties; images are parsed and results are stored automatically.
SQLite Database	All parsed data is stored in properties.db for retrieval and updates.
Smart Chat Assistant	Understands queries like “2BHK under 1500000” or “loan options for 1200000” and returns relevant results.
RESTful API	Exposes endpoints for integration with Streamlit or other front-end dashboards.



Tech Stack

Backend Framework: FastAPI (Python 3.10)

Model Framework: Ultralytics YOLOv8

OCR Engines: Tesseract OCR, EasyOCR (fallback)

Database: SQLite

Other Libraries: OpenCV, Pandas, Pillow, NumPy




📁 Project Structure
property_bot_project/
│
├── backend/
│   ├── app.py                  
│   ├── models/
│   │   └── best_scratch.pt     
│   ├── properties.db           
│   ├── Property_list.xlsx      
│   ├── requirements.txt        
│   └── venv/                   
│
└── README.md                   



Setup & Installation


 Clone or extract the backend folder
cd property_bot_project/backend



Create a virtual environment
python -m venv venv
venv\Scripts\activate


 Install dependencies
pip install -r requirements.txt




If you don’t have a requirements.txt, install manually:

pip install fastapi "uvicorn[standard]" ultralytics opencv-python-headless pytesseract easyocr pandas openpyxl



Ensure Tesseract OCR is installed

Download from: Tesseract for Windows

Then verify:

tesseract --version



Run the server
uvicorn app:app --reload --port 8000



API Endpoints
Method	Endpoint	Description
POST	/parse-floorplan	Upload an image → Detect + OCR → Return structured JSON
POST	/ingest	Upload Excel file of properties for batch parsing
GET	/properties	List all stored property records
POST	/chat	Query with text like “2BHK under 1500000” or “loan for 1200000”
Docs	/docs	Interactive Swagger UI to test endpoints



 Inference Logic

YOLO detects boxes: room names, dimensions, ceilings, furniture, etc.

OCR reads inside each bounding box (both Tesseract + EasyOCR fallback).

Dimensions like 12-6 x 15-8 are converted → sq. ft. area automatically.

Total area and room counts are inferred and updated into properties.db.



Smart Loan Suggestion

The backend can calculate EMI offers dynamically:

For any query like “loan for 1200000”,
returns sample EMI breakdown from multiple banks with varying interest rates.



 Example Output
{
  "counts": {"room_dim": 12, "clg": 3, "things": 8},
  "rooms_detail": [
    {
      "label": "room_dim",
      "bbox": [474, 293, 601, 306],
      "ocr": "11-6 x 11-0",
      "ocr_conf": 90.7,
      "dimension": {
        "raw": "11-6x11-0",
        "w": 11.5,
        "h": 11.0,
        "area_sqft": 126.5
      },
      "conf": 0.88
    }
  ]
}



Sample SQL Query (to verify results)

Run in Python shell:

import sqlite3
conn = sqlite3.connect("properties.db")
rows = conn.execute("SELECT id, bedrooms, bathrooms, area_sqft FROM properties").fetchall()
for r in rows:
    print(r)
conn.close()




Future Enhancements

Add multilingual OCR for Indian regional languages

Integrate Streamlit dashboard for interactive visualization

Expand dataset with more property types

Cloud deployment with AWS Lambda or Render



Name: Satvik Anand
