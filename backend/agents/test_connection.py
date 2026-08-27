from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client()
print("Sending request...")
r = client.interactions.create(model="gemini-3.7-flash", input="Say hello in one sentence.")
print(r.output_text)