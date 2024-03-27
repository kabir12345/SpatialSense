from ollama import Client
import json
client = Client(host='http://localhost:11434')
response = client.chat(model='llava:7b-v1.5-q2_K', messages=[
                {
                    'role': 'user',
                    'content': 'What is in the image',
                    'images' : ['/Users/kabir/Downloads/SpatialSense/samples/images/img1_dc1.png'],
                    'stream':False
                },
                ])
response=json.loads(response.json())
print(response['message']['content'])