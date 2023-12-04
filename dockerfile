FROM pytorch/pytorch:latest AS tel-ai-bot

# Set the working directory
RUN mkdir -p /app/generated_images
RUN mkdir -p /app/images
RUN mkdir -p /app/voice
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY ./models/*.py ./models/
COPY ./.docker.env ./.env
COPY ./*.py .

# Run the application
CMD ["python", "main.py"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
