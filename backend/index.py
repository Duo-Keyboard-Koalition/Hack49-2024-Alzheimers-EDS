from fastapi import FastAPI, File, UploadFile
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

os.makedirs("uploaded_files", exist_ok=True)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Define the path where the file will be saved
    file_path = f"uploaded_files/{file.filename}"
    
    # Open the file in write-binary mode and stream the upload
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # Read in chunks of 1 MB
            if not chunk:
                break
            buffer.write(chunk)
    
    return {"filename": file.filename}

@app.post("/nostream/")
async def create_upload_file_nostream(file: UploadFile = File(...)):
    file_path = f"uploaded_files/{file.filename}"
    
    # Save the uploaded file directly without streaming
    with open(file_path, "wb") as buffer:
        # Read the entire file into memory and write it to the buffer
        content = await file.read()
        buffer.write(content)
    
    return {"filename": file.filename}