FROM pytorch/pytorch:latest AS telegram-ai-bot


# install tools
# RUN apt-get update && apt-get install --assume-yes \
#     nvtop \
#     unzip

# RUN rm -rf /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
#            /usr/lib/x86_64-linux-gnu/libcuda.so*
#     /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
#     /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
#     /usr/local/cuda/compat/lib/*.515.65.01

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .
# Install the dependencies
RUN pip install -r requirements.txt

# Create directories for media files
RUN mkdir -p /app/data/generated_images
RUN mkdir -p /app/data/images
RUN mkdir -p /app/data/voice
RUN mkdir -p /app/data/chat

# Copy the application code
COPY ./models/*.py ./models/
COPY ./*.py ./

# Create dir for storing models
RUN mkdir /root/.cache/huggingface

# Run the application
#CMD ["python", "main.py"]
ENTRYPOINT ["tail", "-f", "/dev/null"]
