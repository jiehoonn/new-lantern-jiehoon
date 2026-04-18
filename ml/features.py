import re
import numpy as np

# ─── Modality ─────────────────────────────────────────────────────────────────

MODALITIES = ['XR', 'CT', 'MRI', 'MR', 'US', 'MAM', 'NM', 'PET', 'ECHO', 'IR', 'DXA', 'FLUORO']

def get_modality(desc):
    desc = desc.upper()
    for m in MODALITIES:
        if re.search(r'\b' + m + r'\b', desc):
            return m
    return 'OTHER'

# ─── Region ───────────────────────────────────────────────────────────────────

REGIONS = [
    ('WHOLE_BODY', ['SKULL TO THIGH', 'PET/CT', 'PET-CT']),
    ('BREAST', ['MAM', 'BREAST', 'MAMMO', 'MAMMOGRAPH', 'MAMMOGRAM', 'ULTRASOUND BILAT SCREEN', 'US BREAST']),
    ('CARDIAC', ['ECHO', 'CARDIAC', 'CORONARY', 'CTA CORONARY', 'CT CORONARY', 'FFR', 'CT FFR', 'MR CARDIAC']),
    ('CHEST', ['CHEST', 'THORAX', 'LUNG', 'PLEURA', 'RIBS', 'RIB', 'NM PUL']),
    ('BRAIN', ['BRAIN', 'HEAD', 'CRANIAL', 'SKULL', 'CEREBRAL', 'STROKE', 'INTRACRAN', 'EEG', 'TRANSCRANIAL']),
    ('NECK', ['NECK', 'THYROID', 'CAROTID', 'SOFT TISSUE NECK', 'LARYNX', 'PAROTID', 'ANGIO HEAD', 'ANGIO CAROTID']),
    ('SPINE_CERVICAL', ['CERVICAL SPINE', 'C-SPINE', 'C SPINE']),
    ('SPINE_LUMBAR', ['LUMBAR', 'L-SPINE', 'L SPINE', 'LUMBOSACRAL']),
    ('SPINE_THORACIC', ['THORACIC SPINE', 'T-SPINE', 'T SPINE']),
    ('SPINE', ['SPINE']),
    ('ABDOMEN', ['ABDOMEN', 'ABDOMINAL', ' ABD', 'ABD/', 'ABD_', 'LIVER', 'PANCREAS', 'KIDNEY', 'RENAL', 'ENTEROGRAPHY', 'UPPER GI', 'BARIUM', 'ESOPHAG', 'BOWEL', 'COLON', 'PELVIS', 'PELVIC']),
    ('HIP', ['HIP']),
    ('KNEE', ['KNEE']),
    ('SHOULDER', ['SHOULDER']),
    ('FOOT', ['FOOT', 'FEET', 'ANKLE', 'TOE', 'CALCAN']),
    ('WRIST', ['WRIST', 'HAND', 'FINGER']),
    ('ELBOW', ['ELBOW']),
    ('LEG', ['FEMUR', 'TIBIA', 'FIBULA', 'LOWER LEG', 'LOWER EXTREM']),
    ('VASCULAR_LEG', [' LE BI', ' LE LT', ' LE RT', 'DOPPLER LE', 'VENOUS IMAGING', 'VENOUS DOPPLER']),
    ('OB_GYN', ['TRANSVAGINAL', 'ENDOVAGINAL', 'OB US', 'FETUS', 'GESTATIONAL', 'OBSTETRIC', 'UTERUS', 'OVARY', 'NUCHAL']),
    ('BLADDER', ['BLADDER', 'URINARY TRACT']),
    ('SINUS', ['SINUS', 'NASAL']),
    ('ORBIT', ['ORBIT', 'OPTIC']),
]

def get_region(desc):
    desc = ' ' + desc.upper() + ' '
    for region, keywords in REGIONS:
        for kw in keywords:
            if kw in desc:
                return region
    return 'OTHER'

# ─── Laterality ───────────────────────────────────────────────────────────────

def get_laterality(desc):
    desc = desc.upper()
    has_rt = bool(re.search(r'\b(RT|RIGHT|RGT)\b', desc))
    has_lt = bool(re.search(r'\b(LT|LEFT|LFT)\b', desc))
    if has_rt and not has_lt: return 'RIGHT'
    if has_lt and not has_rt: return 'LEFT'
    return 'BILATERAL'

# ─── Word overlap ─────────────────────────────────────────────────────────────

def word_overlap(desc1, desc2):
    def words(d):
        return set(re.sub(r'[^a-z0-9 ]', ' ', d.lower()).split())
    return len(words(desc1) & words(desc2))

# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(current_desc, prior_desc, delta_days, tfidf_sim):
    cr = get_region(current_desc)
    pr = get_region(prior_desc)
    cl = get_laterality(current_desc)
    pl = get_laterality(prior_desc)

    same_region = int(cr != 'OTHER' and cr == pr)
    whole_body  = int(cr == 'WHOLE_BODY' or pr == 'WHOLE_BODY')
    same_mod    = int(get_modality(current_desc) == get_modality(prior_desc))
    lat_conflict = int(cl != 'BILATERAL' and pl != 'BILATERAL' and cl != pl)
    overlap     = word_overlap(current_desc, prior_desc)
    log_delta   = np.log1p(delta_days)

    return [
        same_region,
        whole_body,
        same_mod,
        lat_conflict,
        overlap,
        log_delta,
        tfidf_sim,
    ]

FEATURE_NAMES = [
    'same_region',
    'whole_body',
    'same_modality',
    'laterality_conflict',
    'word_overlap',
    'log_delta_days',
    'tfidf_cosine_sim',
]