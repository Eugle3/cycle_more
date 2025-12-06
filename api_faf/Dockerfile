FROM python:3.8.6-buster

COPY app /app
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "app.api_test:app", "--host", "0.0.0.0", "--port", "8000"]
