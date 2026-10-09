FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY app_stock.py .

EXPOSE 8501

CMD ["streamlit", "run", "app_stock.py", "--server.port=8501", "--server.address=0.0.0.0"]