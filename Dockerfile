FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# If any tool (now or future) uses Playwright, uncomment and ensure it's installed:
# RUN python -m playwright install 

COPY . . 
# If 'tools' is a sub-directory of your build context (where Dockerfile is):
# COPY tools ./tools # This line might be covered by 'COPY . .' if tools is in the root

CMD ["python", "proxy.py"]