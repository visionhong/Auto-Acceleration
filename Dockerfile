FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /Auto-Acceleration

RUN git clone https://github.com/visionhong/Auto-Acceleration.git .

RUN pip install -r requirements.txt
