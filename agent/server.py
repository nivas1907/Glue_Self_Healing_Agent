
from fastapi import FastAPI, Request
import uvicorn
from app import run_agent   # your existing analysis function

app = FastAPI()

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    log_text = data["log"]

    agent_output = run_agent(log_text)

    return {"agent_response": agent_output}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
