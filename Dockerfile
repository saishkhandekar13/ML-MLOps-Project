FROM python:3.10-slim

WORKDIR /app

# Copy app code
COPY flask_app/ /app/

# Copy model artifacts
COPY models/model.pkl /app/models/model.pkl
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Copy dependencies file
COPY flask_app/requirements.txt /app/requirements.txt
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local 
# CMD ["python", "app.py"]

# for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]


# this is dockerfile FROM python:3.10-slim

# WORKDIR /app

# COPY flask_app/ /app/

# COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# RUN pip install -r requirements.txt

# RUN python -m nltk.downloader stopwords wordnet

# EXPOSE 5000

# #local
# # CMD ["python", "app.py"]  

# #Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]