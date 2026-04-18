import json
import pickle
import random
import numpy as np
from datetime import date
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from features import extract_features, FEATURE_NAMES

# ─── Load data ────────────────────────────────────────────────────────────────

with open('../relevant_priors_public.json') as f:
    data = json.load(f)

truth_lookup = {
    (t['case_id'], t['study_id']): t['is_relevant_to_current']
    for t in data['truth']
}

pairs = []
for case in data['cases']:
    current = case['current_study']
    for prior in case['prior_studies']:
        label = truth_lookup[(case['case_id'], prior['study_id'])]
        delta_days = (
            date.fromisoformat(current['study_date']) -
            date.fromisoformat(prior['study_date'])
        ).days
        pairs.append({
            'current_desc': current['study_description'],
            'prior_desc': prior['study_description'],
            'label': label,
            'delta_days': delta_days,
            'case_id': case['case_id'],
        })

print(f'Loaded {len(pairs)} pairs ({sum(p["label"] for p in pairs)} relevant)')

# ─── Train/test split at case level ──────────────────────────────────────────

random.seed(42)
all_case_ids = list({case['case_id'] for case in data['cases']})
random.shuffle(all_case_ids)

split = int(len(all_case_ids) * 0.8)
train_ids = set(all_case_ids[:split])
test_ids  = set(all_case_ids[split:])

train_pairs = [p for p in pairs if p['case_id'] in train_ids]
test_pairs  = [p for p in pairs if p['case_id'] in test_ids]

print(f'Train: {len(train_pairs)} pairs  Test: {len(test_pairs)} pairs')

# ─── Fit TF-IDF on training data only ────────────────────────────────────────

train_descs = [p['current_desc'] for p in train_pairs] + [p['prior_desc'] for p in train_pairs]
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), lowercase=True)
vectorizer.fit(train_descs)

def build_features(pair_list):
    current_vecs = vectorizer.transform([p['current_desc'] for p in pair_list])
    prior_vecs   = vectorizer.transform([p['prior_desc']   for p in pair_list])
    sims         = cosine_similarity(current_vecs, prior_vecs).diagonal()
    X = np.array([
        extract_features(p['current_desc'], p['prior_desc'], p['delta_days'], float(sims[i]))
        for i, p in enumerate(pair_list)
    ])
    y = np.array([p['label'] for p in pair_list])
    return X, y

X_train, y_train = build_features(train_pairs)
X_test,  y_test  = build_features(test_pairs)

# ─── Train and evaluate ───────────────────────────────────────────────────────

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = (model.predict(X_train) == y_train).mean()
test_acc  = (model.predict(X_test)  == y_test).mean()

print(f'\nTrain accuracy: {train_acc*100:.2f}%')
print(f'Test accuracy:  {test_acc*100:.2f}%')

# ─── Feature importance ───────────────────────────────────────────────────────

print('\nFeature coefficients (higher = more predictive of relevant):')
for name, coef in sorted(zip(FEATURE_NAMES, model.coef_[0]), key=lambda x: -x[1]):
    print(f'  {name:25s}  {coef:+.3f}')

# ─── Retrain on full data before saving ──────────────────────────────────────

all_descs = [p['current_desc'] for p in pairs] + [p['prior_desc'] for p in pairs]
vectorizer.fit(all_descs)

X_all, y_all = build_features(pairs)
model.fit(X_all, y_all)

print(f'\nFinal train accuracy (full data): {(model.predict(X_all) == y_all).mean()*100:.2f}%')

# ─── Save model artifacts ─────────────────────────────────────────────────────

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

print('Saved model.pkl')