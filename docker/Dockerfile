# C01 推理镜像（Ubuntu 22.04 + CUDA 12.1 + PyTorch 2.3.1；不声明 VOLUME，不依赖运行时下载）
# 构建（联网构建机）:  docker build -f docker/Dockerfile -t c01_em:v2 .
# 导出（离线交付）:    docker save c01_em:v2 -o docker/llm_docker.tar
# 运行:  docker run --gpus all --rm -v $PWD/data:/data c01_em:v2 \
#            --input /data/test.parquet --output /data/pred.jsonl
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 LANG=C.UTF-8
WORKDIR /workspace
COPY environment/requirements_infer.txt /tmp/requirements_infer.txt
RUN pip install --no-cache-dir -r /tmp/requirements_infer.txt
COPY code/ /workspace/code/
COPY weights/config.json weights/model.safetensors weights/param_codec.json /workspace/weights/
RUN mkdir -p /workspace/logs
ENTRYPOINT ["python", "/workspace/code/infer.py", "--weights_dir", "/workspace/weights"]
