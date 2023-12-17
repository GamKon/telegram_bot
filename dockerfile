FROM pytorch/pytorch:latest AS telegram-ai-bot


#install tools
RUN apt-get update && apt-get install --assume-yes \
    curl \
    nano \
    git
#     unzip

#RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
#           /usr/lib/x86_64-linux-gnu/libcuda.so*
#     /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
#     /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
#     /usr/local/cuda/compat/lib/*.515.65.01

# Set the working directory
WORKDIR /app

# Install CrOps Team
ARG OPS_VERSION="2.3.2"
RUN curl -fsSL "https://github.com/nickthecook/crops/releases/download/${OPS_VERSION}/crops.tar.bz2" \
    | tar xj --strip-components=3 -C /usr/local/bin crops/build/linux_x86_64/ops
#Copy ops.yml file
COPY ops_for_image.yml ./ops.yml

# Copy the requirements file
COPY requirements.txt .
# Install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip3 install --upgrade "git+https://github.com/huggingface/transformers" optimum


# Create directories for media files
RUN mkdir -p /app/data/generated_images
RUN mkdir -p /app/data/images
RUN mkdir -p /app/data/voice
RUN mkdir -p /app/data/chat

# Copy the application code
COPY ../models/*.py ./models/
COPY ../*.py ./

# Create dir for storing models
RUN mkdir /root/.cache/huggingface

# Run the application
#CMD ["python", "main.py"]
ENTRYPOINT ["tail", "-f", "/dev/null"]
