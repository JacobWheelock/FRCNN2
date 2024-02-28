# Use an official Python runtime as a parent image
FROM jupyter/datascience-notebook:python-3.11.5

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

ENTRYPOINT ["python3", "-m"]
CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root"]