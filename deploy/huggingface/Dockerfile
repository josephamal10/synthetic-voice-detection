# Hugging Face Space (Docker SDK) for VoiceGuard.
# The Space itself holds only this file and README.md — it builds by cloning
# the public GitHub repo, so GitHub stays the single source of truth.
# After pushing new commits to GitHub: Space → Settings → "Factory rebuild".
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run containers as uid 1000; the app writes users.db and
# analysis_audio/ next to app.py, so the app folder must belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

RUN git clone --depth 1 https://github.com/josephamal10/synthetic-voice-detection.git .
RUN pip install --no-cache-dir --user -r requirements.txt

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# XSRF protection is turned off only here: the Space page embeds the app in
# a cross-site iframe, where Streamlit's XSRF cookie is blocked and every
# file upload would otherwise fail with a 403.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
