FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
COPY ../airflow/dags/models/iris_ann.pkl .
CMD ["python", "app.py"]