from fastapi import FastAPI, File, UploadFile
import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.makedirs("/tmp", exist_ok=True)

app = FastAPI()

# Initialize AWS S3 client
s3_client = boto3.client('s3',
                         region_name='us-east-2',  # Change to your S3 bucket region
                         aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                         aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))

BUCKET_NAME = 'my-ai-models-darcy'

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Define the path where the file will be temporarily saved (optional)
    print("This is create upload function starts")
    temp_file_path = f"/tmp/{file.filename}"  # Temporary storage location

    # Open the file in write-binary mode and stream the upload
    with open(temp_file_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # Read in chunks of 1 MB
            if not chunk:
                break
            buffer.write(chunk)

    # Now upload to S3
    try:
        s3_client.upload_file(temp_file_path, BUCKET_NAME, file.filename)
        print(f'Successfully uploaded {file.filename} to {BUCKET_NAME}')
    except Exception as e:
        print(f'Error uploading file: {e}')

    # Optionally, delete the temporary file after uploading
    os.remove(temp_file_path)

    return {"filename": file.filename}