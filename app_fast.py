import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel


from hate_speech_detection.configuration.config_manager import ConfigurationManager
from hate_speech_detection.pipeline.train_pipeline import TrainPipeline
from hate_speech_detection.pipeline.prediction_pipeline import PredictionPipeline
from hate_speech_detection.exception.exception import CustomException


config_manager = ConfigurationManager()
web_config = config_manager.get_web_config()
app_host = web_config.app_host
app_port = web_config.app_port


app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train", tags=["training"])
async def training():
    async def stream_training_logs():
        try:
            yield "Starting training...\n"
            train_pipeline = TrainPipeline(config_manager)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, train_pipeline.run_pipeline)

            yield "Training successful!\n"
        except Exception as e:
            yield f"Error Occurred! {e}\n"

    return StreamingResponse(stream_training_logs(), media_type="text/plain")


class PredictRequest(BaseModel):
    text: str


@app.post("/predict", tags=["prediction"])
async def predict_route(request: PredictRequest):
    try:
        predict = PredictionPipeline(config_manager)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, predict.run_pipeline, request.text)
        return {"prediction": result}
    except Exception as e:
        raise CustomException(e) from e


if __name__ == "__main__":
    uvicorn.run(app, host=app_host, port=app_port)
