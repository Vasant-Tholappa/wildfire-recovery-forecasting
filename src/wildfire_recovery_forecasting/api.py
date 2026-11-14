"""Local inference only: the hash-verified, frozen Option A XGBoost pipeline."""
from contextlib import asynccontextmanager
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Annotated

import cloudpickle
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "option-a-no-climate-xgboost-2cb3267c3dac4d4c80fe118e274637be"
MODEL_SHA256 = "c4e14763ef726627a5b1e16b354fc71b80d5f41fc800307321c4cddc36aca488"
FEATURES = (
    "nbr_pre_mean", "dominant_evt", "dominant_evt_fraction", "tree_cover_mean",
    "tree_height_mean", "elevation_mean", "slope_mean", "northness_mean", "eastness_mean",
    "fire_size_acres", "ignition_month", "mtbs_unburned_low_fraction", "mtbs_low_fraction",
    "mtbs_moderate_fraction", "mtbs_high_fraction", "mtbs_increased_greenness_fraction",
    "nbr_1_mean", "nbr_drop_patch",
)
CATEGORIES = [
    [3008,3011,3014,3015,3017,3019,3021,3022,3027,3028,3029,3030,3031,3032,
     3033,3034,3043,3044,3045,3053,3058,3061,3062,3112,3113,3114,3118,3151,
     3152,3154,3155,3167,3172,3173,3201,3202,3203,3230,3231,3260,3261,3264,3265,3946],
    [2,4,5,6,7,8,9,10,11,12],
]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
# Tiny reduction roundoff is accepted without modifying values (e.g. 1+2e-16).
Fraction = Annotated[Number, Field(ge=-1e-12, le=1+1e-12)]


class PredictRequest(BaseModel):
    """One patch's original inputs, available through Season 1; no target or IDs."""
    model_config = ConfigDict(extra="forbid")
    nbr_pre_mean: Number
    dominant_evt: Annotated[int, Field(strict=True, gt=0)]
    dominant_evt_fraction: Fraction
    tree_cover_mean: Annotated[Number, Field(ge=0, le=100)]
    tree_height_mean: Annotated[Number, Field(ge=0)]
    elevation_mean: Number
    slope_mean: Annotated[Number, Field(ge=0, le=90)]
    northness_mean: Annotated[Number, Field(ge=-1-1e-12, le=1+1e-12)]
    eastness_mean: Annotated[Number, Field(ge=-1-1e-12, le=1+1e-12)]
    fire_size_acres: Annotated[Number, Field(gt=0)]
    ignition_month: Annotated[int, Field(strict=True, ge=1, le=12)]
    mtbs_unburned_low_fraction: Fraction
    mtbs_low_fraction: Fraction
    mtbs_moderate_fraction: Fraction
    mtbs_high_fraction: Fraction
    mtbs_increased_greenness_fraction: Fraction
    nbr_1_mean: Number
    nbr_drop_patch: Number


class Prediction(BaseModel):
    nbr_5_mean: Annotated[float, Field(allow_inf_nan=False)]
    model_id: str


def load_pipeline(path):
    """Verify bytes BEFORE unpickling; never accept an uploaded model or hash."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
        raise ValueError("Frozen model hash mismatch")
    model = cloudpickle.loads(raw)
    if list(model.feature_names_in_) != list(FEATURES):
        raise ValueError("Frozen feature order mismatch")
    prep = model.named_steps["preprocess"]
    encoder = prep.named_transformers_["categorical"]
    numeric = [f for f in FEATURES if f not in ("dominant_evt", "ignition_month")]
    expected = ["numeric__"+f for f in numeric] + [
        f"categorical__{name}_{value}" for name, values in
        zip(("dominant_evt", "ignition_month"), CATEGORIES) for value in values]
    if (prep.get_feature_names_out().tolist() != expected
            or [c.tolist() for c in encoder.categories_] != CATEGORIES
            or encoder.handle_unknown != "ignore"):
        raise ValueError("Frozen preprocessing mismatch")
    return model


def create_app(loader=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = None
        try:
            app.state.pipeline = (loader or load_pipeline)(os.environ.get("WILDFIRE_MODEL_PATH", ""))
        except Exception:
            # Do not expose filesystem paths, serialized data or exception text.
            logging.getLogger(__name__).error("Frozen model unavailable; verify mount, hash and environment")
        yield
        app.state.pipeline = None

    app = FastAPI(title="After the Fire — NBR5 inference", lifespan=lifespan,
                  description="One patch, Season 1 inputs. NBR5 is spectral condition, not percent recovery.")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Do not echo invalid NaN/Infinity inputs into a JSON response.
        errors = [{key: error[key] for key in ("loc", "msg", "type")}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def ready():
        model = app.state.pipeline
        if model is None:
            raise HTTPException(503, "Frozen model unavailable")
        return model

    @app.get("/health")
    def health():
        ready()
        return {"status": "ready", "model_id": MODEL_ID}

    @app.post("/predict", response_model=Prediction)
    def predict(record: PredictRequest):
        model = ready()
        frame = pd.DataFrame([record.model_dump()], columns=list(FEATURES))
        try:
            values = np.asarray(model.predict(frame))
            if values.shape != (1,) or not math.isfinite(float(values[0])):
                raise ValueError("Invalid model output")
            return Prediction(nbr_5_mean=float(values[0]), model_id=MODEL_ID)
        except Exception:
            raise HTTPException(503, "Prediction unavailable") from None

    return app


app = create_app()
