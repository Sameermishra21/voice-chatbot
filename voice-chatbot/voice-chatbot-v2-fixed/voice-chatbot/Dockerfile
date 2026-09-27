# Container used by Hugging Face Spaces (or Render / Railway / any Docker host).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Safety net: if the trained model was not committed, train it once at BUILD time
# (never at request time).
RUN test -f backend/chatbot_model.keras || python backend/train_model.py

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
