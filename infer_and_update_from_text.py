# infer_and_update_from_text.py
import sqlite3, json, re
from pathlib import Path

DB = "properties.db"

# regexes (case-insensitive)
bhk_re = re.compile(r'(\d+)\s*(?:-?\s*)?(?:bhk|bhk?s|bedroom|bedrooms|br\b)', re.I)
studio_re = re.compile(r'\bstudio\b', re.I)
bath_re = re.compile(r'(\d+)\s*(?:bath|bathroom|bathrooms|toilet|toilets)\b', re.I)
area_re1 = re.compile(r'(\d{2,6}(?:[,\d]{0,})?)\s*(?:sq\.?\s?ft|sqft|sq ft|sq\.ft\.)\b', re.I)
area_re2 = re.compile(r'(\d{2,5})\s*(?:sq\b)', re.I)  # fallback

def extract_from_text(text):
    if not text:
        return None, None, None
    t = str(text)
    bedrooms = None
    bathrooms = None
    area = None

    # bedrooms
    m = bhk_re.search(t)
    if m:
        try:
            bedrooms = int(m.group(1))
        except:
            bedrooms = None
    else:
        if studio_re.search(t):
            bedrooms = 1

    # bathrooms
    mb = bath_re.search(t)
    if mb:
        try:
            bathrooms = int(mb.group(1))
        except:
            bathrooms = None

    # area (sqft)
    ma = area_re1.search(t)
    if ma:
        s = ma.group(1).replace(',', '')
        try:
            area = float(s)
        except:
            area = None
    else:
        m2 = area_re2.search(t)
        if m2:
            try:
                area = float(m2.group(1))
            except:
                area = None

    return bedrooms, bathrooms, area

def main():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    # select rows where bedrooms or area missing (0)
    c.execute("SELECT id,title,price,bedrooms,bathrooms,area_sqft,image_path,parsed_json FROM properties")
    rows = c.fetchall()

    updated = 0
    samples = []
    for r in rows:
        pid, title, price, bedrooms, bathrooms, area, image_path, parsed_json = r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        orig_bed = bedrooms
        orig_bath = bathrooms
        orig_area = area

        # Only attempt if missing (0 or None)
        candidate_text = " ".join(filter(None, [str(title), " ", parsed_json if parsed_json else ""]))
        # parsed_json might be a JSON string with rooms; try long_description by reading parsed_json?
        # but we also want long_description column present in Excel (we didn't store it); try to parse from parsed_json or leave.
        # We'll use title and parsed_json text (which may include long_description if ingestion stored it).
        b, ba, ar = extract_from_text(candidate_text)

        # if not found yet, attempt to search the parsed_json string or metadata tags if present
        if (b is None or ba is None or ar is None) and parsed_json:
            try:
                pj = json.loads(parsed_json)
                # try long_description inside parsed_json? fallback to stringify
                pj_text = json.dumps(pj)
                b2, ba2, ar2 = extract_from_text(pj_text)
                if b is None and b2 is not None:
                    b = b2
                if ba is None and ba2 is not None:
                    ba = ba2
                if ar is None and ar2 is not None:
                    ar = ar2
            except Exception:
                pass

        # final fallback: try keywords in title
        if b is None:
            mm = re.search(r'(\d+)\s*BHK', title, re.I)
            if mm:
                try:
                    b = int(mm.group(1))
                except:
                    b = None

        # apply updates if we inferred something useful
        updates = []
        params = []
        if b is not None and (not bedrooms or int(bedrooms) == 0):
            updates.append("bedrooms = ?")
            params.append(int(b))
        if ba is not None and (not bathrooms or int(bathrooms) == 0):
            updates.append("bathrooms = ?")
            params.append(int(ba))
        if ar is not None and (not area or float(area) == 0.0):
            updates.append("area_sqft = ?")
            params.append(float(ar))

        if updates:
            params.append(pid)
            sql = "UPDATE properties SET " + ", ".join(updates) + " WHERE id = ?"
            try:
                c.execute(sql, params)
                updated += 1
                samples.append((pid, title, updates, params[:-1]))
            except Exception as e:
                print("update error for", pid, e)

    conn.commit()
    conn.close()

    print("Done. Rows scanned:", len(rows))
    print("Rows updated:", updated)
    print("Sample updates (first 10):")
    for s in samples[:10]:
        print(s)

if __name__ == "__main__":
    main()
