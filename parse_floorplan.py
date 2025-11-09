# backend/parse_floorplan.py
import os
from pathlib import Path
import cv2, numpy as np, re
from PIL import Image
import pytesseract
import json

# --- MODEL PATH (do NOT change) ---
MODEL_PATH = Path("models/best_scratch.pt")

# If on Windows and tesseract is not in PATH, set this to your tesseract.exe path:
# Example default Windows install path:
if os.name == "nt":
    # Change if your tesseract is installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Lazy model loader
MODEL = None
def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            MODEL = YOLO(str(MODEL_PATH))
            try:
                print(f"[parse_floorplan] Loaded YOLO model from: {MODEL_PATH.resolve()}")
            except Exception:
                print("[parse_floorplan] Loaded YOLO model (path print failed).")
            return MODEL
        except Exception as e:
            print("Warning: failed to load YOLO model:", e)
            MODEL = None
            return None
    else:
        print(f"[parse_floorplan] Model file not found at {MODEL_PATH.resolve()}")
        return None

# ---------- OCR & postprocessing helpers ----------
# Tesseract options (try different psm values if needed)
DEFAULT_OCR_CONFIG = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\"-x.,:°/"

def preprocess_for_ocr(bgr_crop, resize_max=1600):
    if bgr_crop is None or bgr_crop.size == 0:
        return None
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 6,6,6,6, cv2.BORDER_CONSTANT, value=255)
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h,w) < 300:
        scale = 2.0
    elif max(h,w) < 600:
        scale = 1.5
    neww = min(int(w*scale), resize_max)
    newh = min(int(h*scale), resize_max)
    gray = cv2.resize(gray, (neww, newh), interpolation=cv2.INTER_LINEAR)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return th

def ocr_text_from_crop(bgr_crop, ocr_config=DEFAULT_OCR_CONFIG):
    try:
        img = preprocess_for_ocr(bgr_crop)
        if img is None:
            return "", 0.0
        pil = Image.fromarray(img)
        data = pytesseract.image_to_data(pil, config=ocr_config, output_type=pytesseract.Output.DICT)
        texts = []
        confs = []
        for t, c in zip(data.get('text', []), data.get('conf', [])):
            if t and str(t).strip() and str(c).strip() and int(float(c)) > 0:
                texts.append(str(t).strip())
                try:
                    confs.append(int(float(c)))
                except:
                    pass
        raw = " ".join(texts).strip()
        avg_conf = float(sum(confs)/len(confs)) if confs else 0.0
        raw = re.sub(r"[\s]{2,}", " ", raw)
        return raw, avg_conf
    except Exception as e:
        print("OCR error:", e)
        return "", 0.0

def extract_dimensions(txt):
    if not txt:
        return None
    s = txt.replace(" ", "")
    patterns = [
        r"(\d{1,2}['’]?-?\d{0,2}\"?)\s*[xX×]\s*(\d{1,2}['’]?-?\d{0,2}\"?)",
        r"(\d{1,3}\.?\d?)\s*[xX×]\s*(\d{1,3}\.?\d?)",
        r"(\d{1,3})['’]?[fFtT]?\s*[xX×]\s*(\d{1,3})['’]?"
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(0)
    return None

def expand_bbox(bbox, img_shape, pad=0.05):
    x1,y1,x2,y2 = bbox
    h, w = img_shape[:2]
    dx = int((x2-x1)*pad)
    dy = int((y2-y1)*pad)
    nx1 = max(0, x1-dx)
    ny1 = max(0, y1-dy)
    nx2 = min(w, x2+dx)
    ny2 = min(h, y2+dy)
    return [nx1, ny1, nx2, ny2]

def group_boxes_to_rooms(detections, iou_thresh=0.15):
    boxes = [d['bbox'] for d in detections]
    used = [False]*len(boxes)
    rooms = []
    def iou(a,b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        inter_x1 = max(ax1,bx1); inter_y1 = max(ay1,by1)
        inter_x2 = min(ax2,bx2); inter_y2 = min(ay2,by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
        area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
        return inter / float(area_a + area_b - inter + 1e-8)
    for i,b in enumerate(boxes):
        if used[i]: continue
        group = [i]
        used[i]=True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            if iou(b, boxes[j]) > iou_thresh:
                group.append(j); used[j]=True
        xs = [boxes[k][0] for k in group] + [boxes[k][2] for k in group]
        ys = [boxes[k][1] for k in group] + [boxes[k][3] for k in group]
        merged = [min(xs), min(ys), max(xs), max(ys)]
        rooms.append({"bbox": merged, "children": group})
    return rooms

# ---------- Parsing functions ----------
def parse_with_model(image_path):
    model = get_model()
    if model is None:
        raise RuntimeError("Model not available")
    results = model(image_path)[0]
    img = cv2.imread(image_path)
    raw_dets = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1,y1,x2,y2 = map(int, box)
        if (x2-x1) < 12 or (y2-y1) < 12:
            continue
        label = results.names[int(cls)]
        raw_dets.append({"label": label, "bbox":[x1,y1,x2,y2]})
    # If no detections, fallback to empty
    if not raw_dets:
        return {"counts":{}, "rooms_detail":[]}
    # group into room-level boxes
    room_candidates = group_boxes_to_rooms(raw_dets, iou_thresh=0.12)
    rooms_out = []
    for rc in room_candidates:
        mb = rc['bbox']
        ex = expand_bbox(mb, img.shape, pad=0.08)
        x1,y1,x2,y2 = map(int, ex)
        crop = img[y1:y2, x1:x2]
        text, conf = ocr_text_from_crop(crop)
        dim = extract_dimensions(text)
        child_labels = [raw_dets[i]['label'] for i in rc['children']]
        from collections import Counter
        label = Counter(child_labels).most_common(1)[0][0] if child_labels else "room"
        rooms_out.append({
            "label": label,
            "bbox": [x1,y1,x2,y2],
            "ocr": text,
            "ocr_conf": round(conf,2),
            "dimension": dim,
            "children": [raw_dets[i] for i in rc['children']]
        })
    rooms_out = sorted(rooms_out, key=lambda r: (r['bbox'][1], r['bbox'][0]))
    counts = {}
    for r in rooms_out:
        counts[r['label']] = counts.get(r['label'], 0) + 1
    return {"counts": counts, "rooms_detail": rooms_out}

def parse_floorplan(image_path):
    # Main entry point. Uses the model if available; otherwise returns mock output.
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    try:
        if get_model() is not None:
            return parse_with_model(image_path)
    except Exception as e:
        print("Model parse failed, falling back to mock. Error:", e)
    # MOCK fallback: naive quadrant splits with OCR
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    quads = [
        (0, 0, w//2, h//2),
        (w//2, 0, w, h//2),
        (0, h//2, w//2, h),
        (w//2, h//2, w, h)
    ]
    rooms = []
    counts = {}
    for i,(x1,y1,x2,y2) in enumerate(quads):
        crop = img[y1:y2, x1:x2]
        ocr, conf = ocr_text_from_crop(crop)
        label = "room"
        low = (ocr or "").lower()
        if "kitchen" in low: label = "kitchen"
        elif "bath" in low or "toilet" in low: label = "bathroom"
        elif "bed" in low: label = "bedroom"
        counts[label] = counts.get(label,0) + 1
        rooms.append({"label":label, "bbox":[x1,y1,x2,y2], "ocr":ocr, "ocr_conf": round(conf,2), "dimension": extract_dimensions(ocr)})
    return {"counts": counts, "rooms_detail": rooms}

# If run standalone for debug
if __name__ == "__main__":
    import sys, pprint
    out = parse_floorplan(sys.argv[1])
    pprint.pprint(out)
