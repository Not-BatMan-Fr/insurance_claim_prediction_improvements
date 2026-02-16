FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY environment-canary.yml .

RUN conda env update -n base -f environment-canary.yml && conda clean -a -y

COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/

RUN mkdir -p visualizations

CMD [ "python3", "./src/main.py" ]