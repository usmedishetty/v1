import os
import sys
import json
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [_ROOT, os.path.join(_ROOT, "backend"), os.path.join(_ROOT, "backend", "07_api")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi.testclient import TestClient
from backend.api import app

HEADERS = {"X-API-Key": "super-secret-token"}

def test_extract_endpoint():
    client = TestClient(app)
    demo_pdf_path = os.path.join(_ROOT, "templates", "Land_Acquisition_Project_Data_Sheet_DEMO.pdf")

    print("=== TEST 1: POST /extract-project-pdf WITH DEMO PDF ===")
    with open(demo_pdf_path, "rb") as f:
        files = {"file": ("Land_Acquisition_Project_Data_Sheet_DEMO.pdf", f, "application/pdf")}
        res = client.post("/extract-project-pdf", headers=HEADERS, files=files)
    print(f"Status: {res.status_code}")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    print(f"Extracted {data['filled_count']}/{data['total_fields']} fields.")
    assert data['fields']['inp-project-id'] == 'NHAI-RJ-2023-0001'
    assert data['fields']['inp-state'] == 'Rajasthan'
    assert data['fields']['inp-district'] == 'Banswara'
    assert data['fields']['inp-cost'] == 1485.37
    assert data['fields']['inp-protest-flag'] is False
    print(">>> Endpoint Test 1 PASSED!\n")

    print("=== TEST 2: AUTH REJECTION (NO API KEY) ===")
    with open(demo_pdf_path, "rb") as f:
        files = {"file": ("demo.pdf", f, "application/pdf")}
        res_no_auth = client.post("/extract-project-pdf", files=files)
    print(f"Status: {res_no_auth.status_code}")
    assert res_no_auth.status_code in [401, 403], f"Expected 401/403, got {res_no_auth.status_code}"
    print(">>> Endpoint Test 2 PASSED: Correctly blocked unauthorized request!\n")

    print("=== TEST 3: BLANK TEMPLATE (EXPECT 400) ===")
    blank_pdf_path = os.path.join(_ROOT, "templates", "Form_LA-7_Blank_Template.pdf")
    with open(blank_pdf_path, "rb") as f:
        files = {"file": ("blank.pdf", f, "application/pdf")}
        res_blank = client.post("/extract-project-pdf", headers=HEADERS, files=files)
    print(f"Status: {res_blank.status_code}, detail: {res_blank.json().get('detail')}")
    assert res_blank.status_code == 400
    print(">>> Endpoint Test 3 PASSED: Correctly returned 400 for blank template!\n")

    print("=== TEST 4: UNRELATED PDF (EXPECT 400) ===")
    unrelated_pdf_path = os.path.join(_ROOT, "templates", "unrelated_document.pdf")
    with open(unrelated_pdf_path, "rb") as f:
        files = {"file": ("unrelated.pdf", f, "application/pdf")}
        res_unrelated = client.post("/extract-project-pdf", headers=HEADERS, files=files)
    print(f"Status: {res_unrelated.status_code}, detail: {res_unrelated.json().get('detail')}")
    assert res_unrelated.status_code == 400
    print(">>> Endpoint Test 4 PASSED: Correctly returned 400 for unrelated document!\n")

    print("=== TEST 5: GET /download-blank-template ===")
    res_dl = client.get("/download-blank-template")
    print(f"Status: {res_dl.status_code}, Content-Type: {res_dl.headers.get('Content-Type')}, Size: {len(res_dl.content)} bytes")
    assert res_dl.status_code == 200
    assert len(res_dl.content) > 1000
    print(">>> Endpoint Test 5 PASSED: Blank template downloaded successfully!\n")

if __name__ == "__main__":
    test_extract_endpoint()
