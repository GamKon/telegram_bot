FROM pytorch/pytorch:latest AS telegram-ai-bot

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .
# Install the dependencies
RUN pip install -r requirements.txt

# Create directories for media files
RUN mkdir -p /app/generated_images
RUN mkdir -p /app/images
RUN mkdir -p /app/voice

# Copy the application code
COPY ./models/*.py ./models/
COPY ./*.py ./

# Create dir for storing models
RUN mkdir /root/.cache/huggingface

# Run the application
#CMD ["python", "main.py"]
ENTRYPOINT ["tail", "-f", "/dev/null"]
