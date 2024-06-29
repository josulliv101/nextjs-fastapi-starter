from dotenv import load_dotenv
import os
from fastapi import FastAPI

# from api.graph_chain import CYPHER_GENERATION_PROMPT, graph_chain

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

@app.get("/api/python")
def hello_world():
    return {"message": os.getenv("NEO4J_URI")}