import pickle
from datetime import date
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from ml.features import extract_features

# ─── Load model artifacts ─────────────────────────────────────────────────────

with open('ml/model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model      = artifacts['model']
vectorizer = artifacts['vectorizer']

# ─── Request / response schemas ───────────────────────────────────────────────

class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class Case(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: Study
    prior_studies: List[Study]

class PredictRequest(BaseModel):
    challenge_id: str
    schema_version: int
    cases: List[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class PredictResponse(BaseModel):
    predictions: List[Prediction]

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI()

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    predictions = []

    for case in request.cases:
        current = case.current_study
        priors  = case.prior_studies

        if not priors:
            continue

        # Compute TF-IDF cosine similarity for all priors in one batch
        current_vec = vectorizer.transform([current.study_description])
        prior_vecs = vectorizer.transform([p.study_description for p in priors])
        sims = cosine_similarity(current_vec, prior_vecs)[0]

        for prior, sim in zip(priors, sims):
            delta_days = (
                date.fromisoformat(current.study_date) -
                date.fromisoformat(prior.study_date)
            ).days

            features = extract_features(
                current.study_description,
                prior.study_description,
                delta_days,
                float(sim),
            )

            pred = bool(model.predict([features])[0])

            predictions.append(Prediction(
                case_id=case.case_id,
                study_id=prior.study_id,
                predicted_is_relevant=pred,
            ))

    case_count  = len(request.cases)
    prior_count = len(predictions)
    print(f'[predict] cases={case_count} priors={prior_count}')

    return PredictResponse(predictions=predictions)