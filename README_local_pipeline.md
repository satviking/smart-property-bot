# Local Pipeline (property_bot_project) - README

## Overview
This local pipeline supports:
- Excel ingestion (`/ingest`)
- Floorplan parsing (`/parse-floorplan`) — works with a trained YOLOv8 segmentation/detect model when `backend/models/best_scratch.pt` is present. If not present, the endpoint returns a stable mock output so the rest of the system is testable.
- Property chatbot `/chat` — simple interactive bot that suggests properties and loan offers.

## How to run locally (fast)
1. Create virtualenv and install:
