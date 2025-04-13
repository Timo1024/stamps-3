FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages if specified
ARG ADDITIONAL_PACKAGES=""
RUN if [ -n "$ADDITIONAL_PACKAGES" ]; then pip install --no-cache-dir $ADDITIONAL_PACKAGES; fi

COPY . .

ENV PYTHONPATH=/app

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Use the entrypoint script instead of directly running uvicorn
CMD ["./entrypoint.sh"]
