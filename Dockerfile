FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Copy the current directory contents into the container at /app
COPY . /app


CMD ["bash", "startup.sh"]


