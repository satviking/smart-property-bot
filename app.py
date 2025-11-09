# app.py (final updated file for submission)
import os
import re
import uuid
import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Any

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Configure Tesseract path (update if your install differs) ---
# Windows typical path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- YOLO model load (ultralytics) ---
model = None
try:
    from ultralytics import YOLO
    MODEL_PATH = Path("models/best_scratch.pt")   # <- force this filename
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print(f"[parse_floorplan] Loaded YOLO model from: {MODEL_PATH}")
    else:
        print(f"[parse_floorplan] Model file not found at {MODEL_PATH}")
except Exception as e:
    print("[parse_floorplan] Could not import/load YOLO model:", e)
    model = None

# --- Optional EasyOCR fallback (used if Tesseract confidence is low) ---
EASYOCR_READER = None
try:
    import easyocr

    # initialize CPU reader (fast to import but create once)
    EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    print("[OCR] EasyOCR available for fallback.")
except Exception:
    EASYOCR_READER = None

# --- FastAPI init ---
app = FastAPI(title="Smart Property Bot Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "properties.db"

# ---------- DB setup ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS properties (
        id TEXT PRIMARY KEY,
        title TEXT,
        price REAL,
        bedrooms INTEGER,
        bathrooms INTEGER,
        area_sqft REAL,
        image_path TEXT,
        parsed_json TEXT
    )
    """
    )
    conn.commit()
    conn.close()


init_db()

# ---------- OCR helpers (improved) ----------


def preprocess_for_ocr(crop: np.ndarray) -> Optional[np.ndarray]:
    """Resize/denoise/contrast-enhance crop for OCR (better for tiny text)."""
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Upscale more aggressively for very small crops
    scale = 1.0
    if max(h, w) < 80:
        scale = 4.0
    elif max(h, w) < 200:
        scale = 2.5
    elif max(h, w) < 400:
        scale = 1.8
    neww = max(1, int(w * scale))
    newh = max(1, int(h * scale))
    gray = cv2.resize(gray, (neww, newh), interpolation=cv2.INTER_CUBIC)

    # CLAHE - contrast limited adaptive histogram equalization
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # denoise
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    # adaptive threshold to handle non-uniform background
    try:
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9
        )
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological open to remove speckles and close to join characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    try:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    except Exception:
        pass

    return th


def ocr_text_from_crop_tesseract(crop: np.ndarray) -> (str, float):
    """Try several Tesseract psm settings and return best (text, avg_conf 0..100)."""
    proc = preprocess_for_ocr(crop)
    if proc is None:
        return "", 0.0

    best_text = ""
    best_conf = 0.0
    # try a few PSM modes that often help with floorplan snippets
    psms = [6, 11, 7]  # 6 = block, 11 = sparse, 7 = single line
    for psm in psms:
        try:
            config = (
                f"--psm {psm} -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz"
                f"ABCDEFGHIJKLMNOPQRSTUVWXYZ'\"-x.,:°/\\-"
            )
            data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config=config)
            texts = []
            confs = []
            for t, c in zip(data.get("text", []), data.get("conf", [])):
                if t and str(t).strip():
                    try:
                        ci = int(float(c))
                    except Exception:
                        ci = None
                    if ci is not None and ci > 0:
                        texts.append(str(t).strip())
                        confs.append(ci)
            raw = " ".join(texts).strip()
            avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
            if avg_conf > best_conf and raw:
                best_conf = avg_conf
                best_text = raw
        except Exception:
            continue

    # fallback: if nothing found, return basic string read (low confidence)
    if best_conf == 0.0:
        try:
            raw = pytesseract.image_to_string(proc, config="--psm 6")
            return raw.strip(), 0.0
        except Exception:
            return "", 0.0
    return best_text, best_conf


def ocr_text_from_crop_easyocr(crop: np.ndarray) -> (str, float):
    """Run EasyOCR on crop and return (text, avg_conf_percent)."""
    if EASYOCR_READER is None or crop is None:
        return "", 0.0
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        results = EASYOCR_READER.readtext(img)
        texts = [r[1] for r in results]
        confs = [r[2] for r in results if isinstance(r[2], (int, float))]
        raw = " ".join(texts).strip()
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        # EasyOCR confidences are 0..1, convert to percent-like scale
        return raw, avg_conf * 100.0
    except Exception:
        return "", 0.0


def ocr_text_from_crop(crop: np.ndarray) -> (str, float):
    """Try Tesseract first; if low confidence and EasyOCR available, fallback."""
    txt, conf = ocr_text_from_crop_tesseract(crop)
    # conf is 0..100 (avg percent). We'll accept >=45 as decent.
    if (conf >= 45.0) or EASYOCR_READER is None:
        return txt, conf
    # fallback to EasyOCR
    txt2, conf2 = ocr_text_from_crop_easyocr(crop)
    if txt2 and (conf2 > conf):
        return txt2, conf2
    return txt, conf


def expand_bbox(bbox: List[int], img_shape: Any, pad: float = 0.12) -> List[int]:
    """Expand bbox by pad fraction inside image bounds. Small boxes get larger pad."""
    h_img, w_img = img_shape[:2]
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    # if small area, expand more
    if w * h < 2000:
        pad_use = max(0.25, pad * 2.0)
    elif w * h < 8000:
        pad_use = max(0.18, pad * 1.5)
    else:
        pad_use = pad
    dx = int(w * pad_use)
    dy = int(h * pad_use)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w_img, x2 + dx)
    ny2 = min(h_img, y2 + dy)
    return [nx1, ny1, nx2, ny2]


def parse_dimension_text(txt: str) -> Optional[dict]:
    """Parse many common dimension patterns and return structured info."""
    if not txt:
        return None
    s = txt.replace(" ", "").replace("X", "x").replace("×", "x").replace("|", "")
    # candidate regex patterns (cover formats like 18-10x15-6, 12x10, 12.5 x 10)
    patterns = [
        r"(\d{1,2}[-']\d{1,2}[xX×]\d{1,2}[-']\d{1,2})",
        r"(\d{1,3}\.?\d?)\s*[xX×]\s*(\d{1,3}\.?\d?)",
        r"(\d{1,3}[-']\d{1,2})[xX×](\d{1,3}[-']\d{1,2})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            found = m.group(0)
            # convert forms like 18-10 -> 18 + 10/12
            def to_decimal(x):
                x = x.strip().replace('"', "").replace("'", "-")
                if "-" in x:
                    parts = x.split("-")
                    try:
                        return float(parts[0]) + float(parts[1]) / 12.0
                    except Exception:
                        return None
                try:
                    return float(x)
                except Exception:
                    return None
            parts = re.split(r"[xX×]", found)
            if len(parts) == 2:
                a = to_decimal(parts[0])
                b = to_decimal(parts[1])
                if a and b:
                    return {"raw": found, "w": a, "h": b, "area_sqft": round(a * b, 2)}
            return {"raw": found}
    return None


# ---------- Core floorplan parser ----------
OCR_LABELS = {"room_dim", "hor_dim", "ver_dim", "room_name", "floor_name", "things", "clg"}


def parse_floorplan(image_path: str) -> dict:
    """Run YOLO + OCR on a floorplan image and return parsed JSON."""
    if not model:
        raise RuntimeError("YOLO model not loaded")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model(image_path)
    detections = []
    # results may contain multiple result objects; iterate
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and hasattr(boxes, "xyxy"):
            xy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            # names mapping can be in result.names or model.names
            names_map = None
            try:
                if hasattr(r, "names") and r.names:
                    names_map = r.names
            except Exception:
                names_map = None
            if not names_map and hasattr(model, "names"):
                try:
                    names_map = getattr(model, "names", None)
                except Exception:
                    names_map = None
            for b, cf, cl in zip(xy, confs, cls):
                x1, y1, x2, y2 = map(int, b)
                label = str(int(cl))
                try:
                    if isinstance(names_map, dict):
                        # keys may be ints or strings
                        label = names_map.get(int(cl), names_map.get(str(int(cl)), str(int(cl))))
                    else:
                        # if names_map not dict, leave numeric
                        label = str(int(cl))
                except Exception:
                    label = str(int(cl))
                detections.append({"label": label, "bbox": [x1, y1, x2, y2], "conf": float(cf)})

    parsed_rooms = []
    counts = {}
    for d in detections:
        lbl = d["label"]
        bbox = d["bbox"]
        x1, y1, x2, y2 = expand_bbox(bbox, img.shape, pad=0.12)
        crop = img[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else None

        ocr_text = ""
        ocr_conf = 0.0
        dimension = None

        # run OCR for most labels — cheap and helps capture numbers/names
        if crop is not None:
            ocr_text, ocr_conf = ocr_text_from_crop(crop)
            dimension = parse_dimension_text(ocr_text)

        parsed_rooms.append(
            {
                "label": lbl,
                "bbox": [x1, y1, x2, y2],
                "ocr": ocr_text,
                "ocr_conf": ocr_conf,
                "dimension": dimension,
                "conf": d.get("conf", 0.0),
            }
        )
        counts[lbl] = counts.get(lbl, 0) + 1

    return {"counts": counts, "rooms_detail": parsed_rooms}


# ---------- Inference helper: infer bedrooms/bathrooms/area from parsed JSON ----------
def infer_rooms_from_parsed(parsed: dict) -> dict:
    """
    parsed: dict returned by parse_floorplan()
    Returns dict with inferred fields: bedrooms (int), bathrooms (int), area_sqft (float or None)
    """
    bedrooms = 0
    bathrooms = 0
    total_area = 0.0
    area_found = False

    room_name_texts = []
    for item in parsed.get("rooms_detail", []):
        lbl = (item.get("label") or "").lower()
        ocr = (item.get("ocr") or "").lower()
        if ocr:
            room_name_texts.append(ocr)
        dim = item.get("dimension")
        if isinstance(dim, dict) and dim.get("area_sqft"):
            try:
                total_area += float(dim["area_sqft"])
                area_found = True
            except Exception:
                pass

    bed_keywords = ["bed", "bedroom", "bdrm", "master", "bhk", "guest"]
    bath_keywords = ["bath", "toilet", "wc", "washroom", "bathroom"]

    for t in room_name_texts:
        for bk in bed_keywords:
            if bk in t:
                bedrooms += 1
                break
        for bk in bath_keywords:
            if bk in t:
                bathrooms += 1
                break

    if bedrooms == 0:
        room_name_boxes = sum(1 for it in parsed.get("rooms_detail", []) if (it.get("label") or "").lower() == "room_name")
        if room_name_boxes > 0:
            bedrooms = min(room_name_boxes, max(1, room_name_boxes // 2))

    if bathrooms == 0:
        if any((it.get("label") or "").lower() == "room_name" for it in parsed.get("rooms_detail", [])):
            bathrooms = max(1, bathrooms)

    if not area_found:
        total_area = None
    else:
        total_area = round(total_area, 2)

    try:
        bedrooms = int(bedrooms)
    except:
        bedrooms = 0
    try:
        bathrooms = int(bathrooms)
    except:
        bathrooms = 0

    return {"bedrooms": bedrooms, "bathrooms": bathrooms, "area_sqft": total_area}


# ---------- API Endpoints ----------
@app.post("/parse-floorplan")
async def parse_floorplan_endpoint(file: UploadFile = File(...), property_id: Optional[str] = Form(None)):
    tmp_filename = f"temp_{uuid.uuid4().hex}{Path(file.filename).suffix}"
    tmp_path = Path(tmp_filename)
    with tmp_path.open("wb") as f:
        f.write(await file.read())

    try:
        result = parse_floorplan(str(tmp_path))

        # If property_id provided: infer numeric fields and update DB
        if property_id:
            inferred = infer_rooms_from_parsed(result)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT id, bedrooms, bathrooms, area_sqft FROM properties WHERE id=?", (property_id,))
            existing = c.fetchone()
            if existing:
                new_beds = inferred["bedrooms"] if inferred["bedrooms"] and inferred["bedrooms"] > 0 else existing[1]
                new_baths = inferred["bathrooms"] if inferred["bathrooms"] and inferred["bathrooms"] > 0 else existing[2]
                new_area = inferred["area_sqft"] if inferred["area_sqft"] is not None else existing[3]
                c.execute(
                    "UPDATE properties SET parsed_json=?, bedrooms=?, bathrooms=?, area_sqft=? WHERE id=?",
                    (json.dumps(result), new_beds, new_baths, new_area, property_id),
                )
            else:
                c.execute(
                    """INSERT OR REPLACE INTO properties
                    (id,title,price,bedrooms,bathrooms,area_sqft,image_path,parsed_json)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        property_id,
                        f"Imported {property_id}",
                        0.0,
                        inferred["bedrooms"],
                        inferred["bathrooms"],
                        inferred["area_sqft"] or 0.0,
                        "",
                        json.dumps(result),
                    ),
                )
            conn.commit()
            conn.close()

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": "parse_failed", "msg": str(e)}, status_code=500)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    tmp = f"temp_{uuid.uuid4().hex}.xlsx"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    try:
        df = pd.read_excel(tmp)
    except Exception as e:
        os.remove(tmp)
        return JSONResponse({"error": "read_excel_failed", "msg": str(e)}, status_code=400)
    os.remove(tmp)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    inserted = 0
    parse_success = 0
    parse_failed = 0

    for _, row in df.iterrows():
        pid = str(row.get("property_id") or uuid.uuid4().hex)
        title = str(row.get("title") or "")
        price = float(row.get("price") or 0)
        bedrooms = int(row.get("bedrooms") or 0)
        bathrooms = int(row.get("bathrooms") or 0)
        area = float(row.get("area_sqft") or 0)
        image_path = str(row.get("image_file") or "")
        parsed_json = "{}"
        if image_path and Path(image_path).exists():
            try:
                parsed_json = json.dumps(parse_floorplan(image_path))
                parse_success += 1
            except Exception as e:
                parsed_json = json.dumps({"error": "parse_failed", "msg": str(e)})
                parse_failed += 1
        else:
            parsed_json = json.dumps({"error": "no_image", "msg": "image not found"})
        c.execute(
            """
            INSERT OR REPLACE INTO properties
            (id,title,price,bedrooms,bathrooms,area_sqft,image_path,parsed_json)
            VALUES (?,?,?,?,?,?,?,?)
        """,
            (pid, title, price, bedrooms, bathrooms, area, image_path, parsed_json),
        )
        inserted += 1

    conn.commit()
    conn.close()
    return {"status": "ok", "rows_inserted": inserted, "parse_success": parse_success, "parse_failed": parse_failed}


@app.get("/properties")
def list_properties(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id,title,price,bedrooms,bathrooms,area_sqft,image_path,parsed_json FROM properties LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        parsed = {}
        try:
            parsed = json.loads(r[7]) if r[7] else {}
        except:
            parsed = {"error": "bad_parsed_json"}
        out.append(
            {
                "id": r[0],
                "title": r[1],
                "price": r[2],
                "bedrooms": r[3],
                "bathrooms": r[4],
                "area_sqft": r[5],
                "image_path": r[6],
                "parsed": parsed,
            }
        )
    return out


# ---------- Chat & Loans ----------
def suggest_loans(budget: float, tenure_years: int = 20):
    offers = []
    rate = 0.085 if budget <= 500000 else 0.075 if budget <= 2000000 else 0.065
    r = rate / 12
    n = tenure_years * 12
    if r == 0:
        emi = budget / n
    else:
        emi = budget * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    offers.append({"bank": "Demo Bank A", "apr": rate, "tenure_years": tenure_years, "monthly_emi": round(emi, 2)})
    offers.append({"bank": "Demo Bank B", "apr": rate + 0.01, "tenure_years": tenure_years, "monthly_emi": round(emi * 1.03, 2)})
    return offers


def search_properties(min_beds: Optional[int] = None, max_price: Optional[float] = None, limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT id,title,price,bedrooms,bathrooms,area_sqft,image_path,parsed_json FROM properties WHERE 1=1"
    params: List[Any] = []
    if min_beds is not None:
        q += " AND bedrooms >= ?"
        params.append(min_beds)
    if max_price is not None:
        q += " AND price <= ?"
        params.append(max_price)
    q += " LIMIT ?"
    params.append(limit)
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        parsed = {}
        try:
            parsed = json.loads(r[7]) if r[7] else {}
        except:
            parsed = {"error": "bad_parsed_json"}
        out.append({"id": r[0], "title": r[1], "price": r[2], "bedrooms": r[3], "bathrooms": r[4], "area_sqft": r[5], "image_path": r[6], "parsed": parsed})
    return out


@app.post("/chat")
async def chat_endpoint(user_message: str = Form(...), min_bedrooms: Optional[int] = Form(None), max_price: Optional[float] = Form(None)):
    text = user_message.lower()
    # structured filters provided
    if min_bedrooms is not None or max_price is not None:
        props = search_properties(min_beds=min_bedrooms, max_price=max_price)
        return {"type": "property_suggestions", "count": len(props), "properties": props}

    if "loan" in text or "emi" in text or "finance" in text:
        m = re.search(r"(\d[\d,]*)", text.replace(",", ""))
        if m:
            budget = float(m.group(1))
            offers = suggest_loans(budget)
            return {"type": "loan_offers", "budget": budget, "offers": offers}
        else:
            return {"type": "ask_budget", "message": "What is your budget (approx)? Provide a number like 1200000."}

    # keyword extraction (2BHK / under X)
    min_beds = None
    max_price_v = None
    if "2bhk" in text or "2 bhk" in text or "2-bedroom" in text:
        min_beds = 2
    m2 = re.search(r"under\s+(\d[\d,]*)", text.replace(",", ""))
    if m2:
        max_price_v = float(m2.group(1))
    if min_beds or max_price_v:
        props = search_properties(min_beds=min_beds, max_price=max_price_v)
        return {"type": "property_suggestions", "properties": props}

    return {"type": "ask_clarify", "message": "Tell me your requirement briefly (e.g., '2BHK under 1500000' or ask 'loan options for 1200000')."}


# ---------- Startup message ----------
@app.on_event("startup")
def on_startup():
    if model:
        print("[startup] YOLO model loaded successfully.")
    else:
        print("[startup] YOLO not loaded; parse-floorplan endpoint will return error if used.")


# ---------- Entry ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
