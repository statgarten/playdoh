#!/bin/bash

streamlit run app.py & uvicorn server:app --port 8001
