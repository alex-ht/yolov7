FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y zip htop screen libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# pip install required packages
RUN pip install seaborn thop boto3 mlflow==1.24 \
                AIMaker requests mlflow-asus-aimaker
ADD . /yolov7
