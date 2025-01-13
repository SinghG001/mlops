FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN true
RUN pip install -r requirements.txt
COPY . .
CMD ["flask","run"]