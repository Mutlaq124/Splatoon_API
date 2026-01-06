from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import os
import cv2
import numpy as np
import uvicorn

from utils import (
    preprocess_image,
    postprocess_detections,
    draw_boxes,
    load_model,
    infer,
)

load_dotenv()

app = FastAPI(title="Splatoon YOLOv8 Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_int8.onnx")
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))
CLASS_NAMES = ["aim_cursor", "entity", "obstacle"]
ort_session = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global ort_session
    try:
        ort_session = load_model(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Splatoon YOLOv8 Detection API",
        "endpoints": {
            "/health": "Health check",
            "/predict/json": "Get JSON detections only",
            "/predict/image": "Get annotated image file",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if ort_session else "unhealthy",
        "model_path": MODEL_PATH,
    }


@app.post("/predict/json")
async def predict_json(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
):
    """
    Predict objects and return JSON only (no image)
    Faster response for headless scenarios
    """
    if not ort_session:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image, IMG_SIZE)

        outputs = infer(ort_session, input_data)
        detections = postprocess_detections(outputs[0], conf, iou, CLASS_NAMES)

        results = []
        for det in detections:
            results.append(
                {
                    "class": det["class_name"],
                    "box": {
                        "x1": det["bbox"][0],
                        "y1": det["bbox"][1],
                        "x2": det["bbox"][2],
                        "y2": det["bbox"][3],
                    },
                    "confidence": det["confidence"],
                }
            )

        return JSONResponse({"detections": results, "num_detections": len(results)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    line_thickness: int = Query(2, ge=1, le=10),
):
    """
    Predict objects and return annotated image file
    Useful for direct visualization
    """
    if not ort_session:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image, IMG_SIZE)

        outputs = infer(ort_session, input_data)
        detections = postprocess_detections(outputs[0], conf, iou, CLASS_NAMES)

        img_with_boxes = draw_boxes(
            np.array(image), detections, thickness=line_thickness
        )

        # Return as image file
        _, buffer = cv2.imencode(".jpg", img_with_boxes)
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=annotated_{file.filename}"
            },
        )

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
