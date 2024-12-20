from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import boto3
import os
from dotenv import load_dotenv
import torch
import torchaudio
from pydub import AudioSegment
import speech_recognition as sr

from chat import chat
from tts import tts
from model import EncoderDecoder, TimeSeriesClassifier

# Load environment variables from .env file
load_dotenv()

THRESHOLD = 0.5

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
recognizer = sr.Recognizer()

def load_model(model_path):
    # Initialize the model architecture
    input_dim = 29
    hidden_dim = 15
    encoder = bundle.get_model().to(device)
    decoder = TimeSeriesClassifier(input_dim = input_dim, 
                               num_heads = input_dim, 
                               hidden_dim = hidden_dim, 
                               output_dim = 1)
    model = EncoderDecoder(encoder, decoder).to(device)
    
    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(file):
    filename = file.split('.')[0]
    sound = AudioSegment.from_file(file, format="m4a")
    sound.export(f"{filename}.wav", format="wav")

    with sr.AudioFile(filename + ".wav") as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        # gem_resp = chat(text)
        print("Text recognized: ", text)
        # print("Gem:", gem_resp)

        # tts(gem_resp)

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    waveform, sample_rate = torchaudio.load(filename + ".wav")
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    waveform = waveform.to(device)
    print(waveform)
    with torch.no_grad():
        prediction = model(waveform)
        print(prediction)

    prediction = prediction.item()
    if prediction > THRESHOLD:
        gem = chat(text, healthy=False)
    else:
        gem = chat(text)

    tts(gem)

    return prediction

model_path = './hack49_encoder_decoder_model.pth'
model = load_model(model_path)

os.makedirs("/tmp", exist_ok=True)

app = FastAPI()

# Initialize AWS S3 client
s3_client = boto3.client('s3',
                         region_name='us-east-2',  # Change to your S3 bucket region
                         aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                         aws_secret_access_key=os.getenv('AWS_SECRET_KEY'))

BUCKET_NAME = 'my-ai-models-darcy'

def download_from_s3(bucket_name, s3_file_path, local_file_path):
    s3_client.download_file(bucket_name, s3_file_path, local_file_path)
    print(f"File downloaded: {local_file_path}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/uploadfile/")
async def upload_and_classify(file: UploadFile = File(...)):
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

    download_from_s3(BUCKET_NAME, file.filename, file.filename)

    print(file.filename)
    prediction = predict(file.filename)
    return {"prediction": prediction} # return prediction

@app.get("/get-audio")
async def get_audio():
    # Path to your audio file
    audio_file_path = "output.mp3"
    print('audio req received')

    # Check if the file exists
    if os.path.exists(audio_file_path):
        print('responding with audio')
        return FileResponse(path=audio_file_path, media_type='audio/mpeg', filename="output.mp3")
    else:
        return {"error": "Audio file not found"}, 404