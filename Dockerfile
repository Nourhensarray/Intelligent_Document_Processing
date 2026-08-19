FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CORRECTION : Créer tous les liens symboliques (.so) attendus par PaddlePaddle pour l'inférence
# L'image runtime ne contient que les versions numérotées (.so.11, .so.8), on crée donc les liens génériques
RUN ldconfig -p | grep -E 'cudnn|cublas|curand|cusolver|cufft|cusparse' | awk '{print $NF}' | while read path; do \
        dir=$(dirname "$path"); \
        base=$(basename "$path" | cut -d. -f1-2); \
        ln -sf "$path" "$dir/$base"; \
    done

WORKDIR /app

COPY requirements_vm.txt .

RUN python3.11 -m pip install --upgrade pip

# Installer PaddlePaddle (la version standard 2.6.2 détecte correctement CUDA)
RUN python3.11 -m pip install \
    paddlepaddle-gpu==2.6.2

RUN python3.11 -m pip install \
    Flask==3.1.3 \
    gunicorn \
    paddleocr==2.8.1 \
    numpy==1.26.4 \
    opencv-contrib-python==4.10.0.84 \
    opencv-python==4.11.0.86 \
    pillow==12.2.0 \
    RapidFuzz==3.14.5

COPY . .

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "webapp:app"]