# backfill_from_excel.py
"""
Backfill properties.db from Property_list.xlsx.
- Prints Excel columns and sample rows.
- Tries multiple common column name variants for mapping.
- Updates existing DB rows (by property_id or title).
- Optional: runs parse_floorplan on resolved image paths (disabled by default).
"""

import sqlite3, json, uuid, os
from pathlib import Path
import pandas as pd

EXCEL_PATH = Path("Property_list.xlsx")  # adjust if your Excel is elsewhere
PARSE_IMAGES = False   # set True to call parse_floorplan for image files (may be slow)

# variants to try for each field
CANDIDATES = {
    "property_id": ["property_id", "id", "prop_id", "PropertyID", "propertyid"],
    "title": ["title", "name", "property_title", "property", "listing_title"],
    "price": ["price", "amount", "listing_price", "cost", "Price"],
    "bedrooms": ["bedrooms", "beds", "bhk", "BHK", "no_of_bedrooms", "bedrooms_count"],
    "bathrooms": ["bathrooms", "baths", "toilets", "washrooms"],
    "area_sqft": ["area_sqft", "area", "size_sqft", "sqft", "area_in_sqft"],
    "image_file": ["image_file", "image", "image_path", "imagefile", "images"]
}

def find_column(df_cols, candidates):
    for c in candidates:
        for col in df_cols:
            if col.lower().strip() == c.lower().strip():
                return col
    # try contains pattern
    for c in candidates:
        for col in df_cols:
            if c.lower() in col.lower():
                return col
    return None

def main():
    excel_path = EXCEL_PATH
    if not excel_path.exists():
        print("Excel not found at", excel_path.resolve())
        return

    df = pd.read_excel(excel_path)
    print("Excel loaded. Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nMapping columns...")

    mapped = {}
    for field, cand in CANDIDATES.items():
        col = find_column(df.columns, cand)
        mapped[field] = col
        print(f"  {field} -> {col}")

    conn = sqlite3.connect("properties.db")
    c = conn.cursor()

    updates = 0
    inserted_missing = 0
    for idx, row in df.iterrows():
        # extract values safely
        pid = str(row.get(mapped["property_id"]) or "").strip()
        title = str(row.get(mapped["title"]) or "").strip()
        price = row.get(mapped["price"]) if mapped["price"] else None
        bedrooms = row.get(mapped["bedrooms"]) if mapped["bedrooms"] else None
        bathrooms = row.get(mapped["bathrooms"]) if mapped["bathrooms"] else None
        area = row.get(mapped["area_sqft"]) if mapped["area_sqft"] else None
        image_raw = row.get(mapped["image_file"]) if mapped["image_file"] else None

        # normalize numeric values
        try:
            price = float(price) if price is not None and str(price).strip() != "" else None
        except:
            # strip commas if present
            try:
                price = float(str(price).replace(",",""))
            except:
                price = None
        try:
            bedrooms = int(bedrooms) if bedrooms not in (None, "", float("nan")) else None
        except:
            try:
                bedrooms = int(float(bedrooms))
            except:
                bedrooms = None
        try:
            bathrooms = int(bathrooms) if bathrooms not in (None, "", float("nan")) else None
        except:
            try:
                bathrooms = int(float(bathrooms))
            except:
                bathrooms = None
        try:
            area = float(area) if area not in (None, "", float("nan")) else None
        except:
            area = None

        # find existing DB row by property_id or title
        db_row = None
        if pid:
            c.execute("SELECT id FROM properties WHERE id = ?", (pid,))
            db_row = c.fetchone()
        if not db_row and title:
            c.execute("SELECT id FROM properties WHERE title = ?", (title,))
            db_row = c.fetchone()

        if db_row:
            db_id = db_row[0]
            # build update statement only for non-None fields
            updates_list = []
            params = []
            if price is not None:
                updates_list.append("price = ?"); params.append(price)
            if bedrooms is not None:
                updates_list.append("bedrooms = ?"); params.append(bedrooms)
            if bathrooms is not None:
                updates_list.append("bathrooms = ?"); params.append(bathrooms)
            if area is not None:
                updates_list.append("area_sqft = ?"); params.append(area)
            if image_raw and str(image_raw).strip():
                # resolve image path attempts
                im = str(image_raw).strip()
                # try absolute
                if os.path.isabs(im) and os.path.exists(im):
                    image_path = im
                else:
                    cand = Path("images") / im
                    if cand.exists():
                        image_path = str(cand.resolve())
                    else:
                        cand2 = Path(im)
                        if cand2.exists():
                            image_path = str(cand2.resolve())
                        else:
                            image_path = ""
                if image_path:
                    updates_list.append("image_path = ?"); params.append(image_path)
            if updates_list:
                params.append(db_id)
                sql = "UPDATE properties SET " + ", ".join(updates_list) + " WHERE id = ?"
                c.execute(sql, params)
                updates += 1
                # optional: parse image now and update parsed_json
                if PARSE_IMAGES and image_path:
                    try:
                        from parse_floorplan import parse_floorplan
                        parsed = parse_floorplan(image_path)
                        c.execute("UPDATE properties SET parsed_json = ? WHERE id = ?", (json.dumps(parsed), db_id))
                    except Exception as e:
                        print("warning: parse failed for", image_path, "->", e)
        else:
            # no matching row; insert as new property (optional)
            new_id = pid or str(uuid.uuid4().hex)
            parsed_json = json.dumps({"error":"no_image","msg":"not parsed"})
            c.execute('INSERT OR REPLACE INTO properties (id,title,price,bedrooms,bathrooms,area_sqft,image_path,parsed_json) VALUES (?,?,?,?,?,?,?,?)',
                      (new_id, title or "", price or 0.0, bedrooms or 0, bathrooms or 0, area or 0.0, "", parsed_json))
            inserted_missing += 1

    conn.commit()
    conn.close()

    print(f"Done. Updated rows: {updates}, Inserted new rows: {inserted_missing}")
    print("If updates==0, check the mapped columns above; you can edit CANDIDATES or Excel column names.")
    print("Backup is at properties.db.bak if you need to revert.")
    print("Finished.")
    
if __name__ == '__main__':
    main()
