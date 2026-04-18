const express = require('express')
const app = express()
const port = process.env.PORT || 3000

app.use(express.json({ limit: '10mb' }))

// ─── Region classification ────────────────────────────────────────────────────
// Ordered list: first match wins. More specific entries should come first.
const REGIONS = [
  ['WHOLE_BODY',       ['SKULL TO THIGH', 'PET/CT', 'PET-CT']],
  ['BREAST',           ['MAM', 'BREAST', 'MAMMO', 'MAMMOGRAPH', 'MAMMOGRAM', 'ULTRASOUND BILAT SCREEN', 'US BREAST']],
  ['CARDIAC',          ['ECHO', 'CARDIAC', 'CORONARY', 'CTA CORONARY', 'CT CORONARY', 'FFR', 'CT FFR', 'MR CARDIAC']],
  ['CHEST',            ['CHEST', 'THORAX', 'LUNG', 'PLEURA', 'RIBS', 'RIB', 'NM PUL']],
  ['BRAIN',            ['BRAIN', 'HEAD', 'CRANIAL', 'SKULL', 'CEREBRAL', 'STROKE', 'INTRACRAN', 'EEG', 'TRANSCRANIAL']],
  ['NECK',             ['NECK', 'THYROID', 'CAROTID', 'SOFT TISSUE NECK', 'LARYNX', 'PAROTID', 'ANGIO HEAD', 'ANGIO CAROTID']],
  ['SPINE_CERVICAL',   ['CERVICAL SPINE', 'C-SPINE', 'C SPINE']],
  ['SPINE_LUMBAR',     ['LUMBAR', 'L-SPINE', 'L SPINE', 'LUMBOSACRAL']],
  ['SPINE_THORACIC',   ['THORACIC SPINE', 'T-SPINE', 'T SPINE']],
  ['SPINE',            ['SPINE']],
  ['ABDOMEN',          ['ABDOMEN', 'ABDOMINAL', ' ABD', 'ABD/', 'ABD_', 'LIVER', 'PANCREAS', 'KIDNEY', 'RENAL', 'ENTEROGRAPHY', 'UPPER GI', 'BARIUM', 'ESOPHAG', 'BOWEL', 'COLON', 'PELVIS', 'PELVIC']],
  ['HIP',              ['HIP']],
  ['KNEE',             ['KNEE']],
  ['SHOULDER',         ['SHOULDER']],
  ['FOOT',             ['FOOT', 'FEET', 'ANKLE', 'TOE', 'CALCAN']],
  ['WRIST',            ['WRIST', 'HAND', 'FINGER']],
  ['ELBOW',            ['ELBOW']],
  ['LEG',              ['FEMUR', 'TIBIA', 'FIBULA', 'LOWER LEG', 'LOWER EXTREM']],
  ['VASCULAR_LEG',     [' LE BI', ' LE LT', ' LE RT', 'DOPPLER LE', 'VENOUS IMAGING', 'VENOUS DOPPLER']],
  ['OB_GYN',           ['TRANSVAGINAL', 'ENDOVAGINAL', 'OB US', 'FETUS', 'GESTATIONAL', 'OBSTETRIC', 'UTERUS', 'OVARY', 'NUCHAL']],
  ['BLADDER',          ['BLADDER', 'URINARY TRACT']],
  ['SINUS',            ['SINUS', 'NASAL']],
  ['ORBIT',            ['ORBIT', 'OPTIC']],
]

// Cross-region pairs that are clinically relevant to each other.
// Validated on public eval: only ABDOMEN/OB_GYN has a positive net benefit.
const CROSS_REGION_RELEVANT = new Set([
  'ABDOMEN|OB_GYN',
  'OB_GYN|ABDOMEN',
])

function getRegion(desc) {
  const upper = ' ' + desc.toUpperCase() + ' '
  for (const [region, keywords] of REGIONS) {
    for (const kw of keywords) {
      if (upper.includes(kw)) return region
    }
  }
  return 'OTHER'
}

function getLaterality(desc) {
  const upper = desc.toUpperCase()
  const hasRight = /\b(RT|RIGHT|RGT)\b/.test(upper)
  const hasLeft = /\b(LT|LEFT|LFT)\b/.test(upper)
  if (hasRight && !hasLeft) return 'RIGHT'
  if (hasLeft && !hasRight) return 'LEFT'
  return 'BILATERAL'
}

function isRelevant(currentDesc, priorDesc) {
  const cr = getRegion(currentDesc)
  const pr = getRegion(priorDesc)

  // Whole-body scans (PET/CT) are always relevant to any current study
  if (cr === 'WHOLE_BODY' || pr === 'WHOLE_BODY') return true

  const sameRegion = cr !== 'OTHER' && cr === pr
  const crossRelevant = CROSS_REGION_RELEVANT.has(`${cr}|${pr}`)

  if (!sameRegion && !crossRelevant) return false

  // Contralateral exams are not relevant (e.g. right knee vs left knee)
  const cl = getLaterality(currentDesc)
  const pl = getLaterality(priorDesc)
  if (cl !== 'BILATERAL' && pl !== 'BILATERAL' && cl !== pl) return false

  return true
}

// ─── Cache ────────────────────────────────────────────────────────────────────
// Keyed by "currentDesc|||priorDesc" — avoids recomputing the same pair on
// retries or repeated study combinations across cases.
const cache = new Map()

function cachedIsRelevant(currentDesc, priorDesc) {
  const key = `${currentDesc}|||${priorDesc}`
  if (cache.has(key)) return cache.get(key)
  const result = isRelevant(currentDesc, priorDesc)
  cache.set(key, result)
  return result
}

// ─── Endpoint ─────────────────────────────────────────────────────────────────
app.post('/predict', (req, res) => {
  const body = req.body
  const caseCount = body?.cases?.length ?? 0
  let priorCount = 0

  if (!body || !Array.isArray(body.cases)) {
    return res.status(400).json({ error: 'Invalid request: missing cases array' })
  }

  const predictions = []

  for (const c of body.cases) {
    const currentDesc = c.current_study?.study_description ?? ''
    const priors = c.prior_studies ?? []
    priorCount += priors.length

    for (const prior of priors) {
      predictions.push({
        case_id: c.case_id,
        study_id: prior.study_id,
        predicted_is_relevant: cachedIsRelevant(currentDesc, prior.study_description ?? ''),
      })
    }
  }

  console.log(`[predict] cases=${caseCount} priors=${priorCount} predictions=${predictions.length}`)

  return res.json({ predictions })
})

app.get('/health', (req, res) => res.json({ status: 'ok' }))

app.listen(port, () => {
  console.log(`Server listening on port ${port}`)
})