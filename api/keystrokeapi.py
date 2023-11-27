from fastapi import FastAPI

## Run python -m uvicorn keystrokeapi:app --reload
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}
