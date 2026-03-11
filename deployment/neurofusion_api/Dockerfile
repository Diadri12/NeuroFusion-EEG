FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir     torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY final_model.pt .
COPY feature_scaler.pkl .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
