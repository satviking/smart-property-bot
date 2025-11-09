import requests, json, sys
url = "http://127.0.0.1:8000/parse-floorplan"
fp = "sample_floorplan.jpg"
try:
    with open(fp, "rb") as f:
        files = {"file": f}
        data = {"property_id": "DEMO-2BHK"}
        r = requests.post(url, files=files, data=data, timeout=120)
        print("status:", r.status_code)
        print(json.dumps(r.json(), indent=2))
except Exception as e:
    print("Error:", e)
    sys.exit(1)
