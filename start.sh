#!/bin/bash
streamlit run app.py & uvicorn server:app --reload --host=0.0.0.0 --port=8001

