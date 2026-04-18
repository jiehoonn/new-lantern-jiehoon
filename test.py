import json
import requests

EVAL_FILE = './relevant_priors_public.json'
ENDPOINT  = 'http://localhost:3000/predict'
CASE_LIMIT = None  # set to a number like 20 to spot-check, None for full eval

with open(EVAL_FILE) as f:
    data = json.load(f)

truth_lookup = {
    (t['case_id'], t['study_id']): t['is_relevant_to_current']
    for t in data['truth']
}

cases = data['cases'] if CASE_LIMIT is None else data['cases'][:CASE_LIMIT]
total_priors = sum(len(c['prior_studies']) for c in cases)
print(f'Sending {len(cases)} cases / {total_priors} priors to {ENDPOINT}')

res = requests.post(ENDPOINT, json={
    'challenge_id': 'relevant-priors-v1',
    'schema_version': 1,
    'cases': cases,
})

if not res.ok:
    print(f'HTTP {res.status_code}: {res.text}')
    exit(1)

predictions = res.json()['predictions']

correct = 0
wrong   = []

for pred in predictions:
    key   = (pred['case_id'], pred['study_id'])
    truth = truth_lookup.get(key)
    if truth is None:
        continue
    if pred['predicted_is_relevant'] == truth:
        correct += 1
    else:
        wrong.append((pred, truth))

total = len(predictions)
print(f'\nResults: {correct}/{total} correct ({correct/total*100:.1f}%)')

if wrong:
    print(f'\nSample incorrect predictions:')
    for pred, truth in wrong[:10]:
        case = next(c for c in cases if c['case_id'] == pred['case_id'])
        prior = next(p for p in case['prior_studies'] if p['study_id'] == pred['study_id'])
        print(f'  predicted={pred["predicted_is_relevant"]} truth={truth}')
        print(f'    current: "{case["current_study"]["study_description"]}"')
        print(f'    prior:   "{prior["study_description"]}"')