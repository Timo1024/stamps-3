FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Use the entrypoint script instead of directly running uvicorn
CMD ["./entrypoint.sh"]
