# Sandbox image for Step 2 — preinstalled deps, built once and reused per run
# (installing pandas/numpy/sklearn fresh on every container start would be slow).
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0

COPY runner.py /runner.py

ENTRYPOINT ["python", "/runner.py"]
