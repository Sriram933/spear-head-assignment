from fastapi import FastAPI, UploadFile, File,HTTPException
from modelHandler import ModelHandler
app=FastAPI()
modelHandler=ModelHandler()




@app.post("/upload")
async def upload(csvFile: UploadFile = File(...)):
    if not csvFile.filename.endswith(".csv"):
        return HTTPException(status_code=400, detail="File must be a CSV file")
    return modelHandler.set_csv(csvFile.file)

@app.post("/train")
async def train():
    return modelHandler.train()
@app.post("/predict")
async def train(json:dict):
    return modelHandler.predict(json)




