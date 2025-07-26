from fastapi import FastAPI, Header, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.logic import process_query

app = FastAPI()

API_KEY = "88ba0f462c7a91e77b3015b09ec8f27aa9b814b581cf83830120a95d12d82c9f"

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def hackrx_run(payload: QueryRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    
    answers = process_query(payload.documents, payload.questions)
    return {"answers": answers}
