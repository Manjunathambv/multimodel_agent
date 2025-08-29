
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss


# Load BLIP (image captioning)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model_img2txt = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")


def image_to_text(img_path):
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model_img2txt.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# Use Hugging Face pipeline (GPT-2, replace with bigger/finetuned as needed)
generator = pipeline('text-generation', model='gpt2')


def text_response(prompt):
    return generator(prompt, max_length=50)[0]['generated_text']


# Simple FAISS-based retrieval from a set of documents/captions
documents = ["This is a cat image.", "A person walking in a park.",
             "A historic building."]  # Example docs
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


def retrieve(query, k=2):
    q_embed = embedder.encode([query])
    D, I = index.search(q_embed, k)
    results = [documents[i] for i in I[0]]
    return "\n".join(results)


def multimodal_agent(img_path, user_text):
    # Step 1: Convert image to text
    caption = image_to_text(img_path)
    # Step 2: Retrieve context related to caption and user text
    context = retrieve(caption + " " + user_text)
    # Step 3: Generate LLM response based on everything
    prompt = f"Context: {context}\nImage Caption: {caption}\nUser Query: {user_text}\nAnswer:"
    response = text_response(prompt)
    return response


result = multimodal_agent(
    "image/its-free-featured.jpg", "Describe the main object in this picture.")
print(result)
