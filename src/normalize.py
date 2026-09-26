"""
src/normalize.py
SageMaker Processing Job — Text Normalization

Reads the 6 raw source TSVs and writes normalized versions to S3.
Can also be imported as a module: from src.normalize import normalize_record

Local test:  python src/normalize.py --input-dir data/ --output-dir output/
As a job:    see notebooks/03_normalization_job.ipynb
"""

import os
import re
import subprocess
import unicodedata
import argparse
import pandas as pd

subprocess.run(['pip', 'install', 'indic-transliteration', '-q'], check=True)

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate as _transliterate


# ── Lookup tables ─────────────────────────────────────────────────────────────

SCRIPT_PATTERNS = {
    'devanagari': r'[\u0900-\u097F]',
    'bengali':    r'[\u0980-\u09FF]',
    'tamil':      r'[\u0B80-\u0BFF]',
    'telugu':     r'[\u0C00-\u0C7F]',
    'gujarati':   r'[\u0A80-\u0AFF]',
    'gurmukhi':   r'[\u0A00-\u0A7F]',
    'kannada':    r'[\u0C80-\u0CFF]',
    'malayalam':  r'[\u0D00-\u0D7F]',
}

INDIC_SCHEMES = {
    'devanagari': sanscript.DEVANAGARI,
    'bengali':    sanscript.BENGALI,
    'gurmukhi':   sanscript.GURMUKHI,
    'gujarati':   sanscript.GUJARATI,
    'tamil':      sanscript.TAMIL,
    'telugu':     sanscript.TELUGU,
    'kannada':    sanscript.KANNADA,
    'malayalam':  sanscript.MALAYALAM,
}

# Abbreviated form is canonical. Multi-word phrases listed first so they
# match before single-word entries during normalization.
LEGAL_SUFFIXES = {
    'private limited': 'pvt ltd',
    'pvt. ltd.': 'pvt ltd',  'pvt. ltd': 'pvt ltd',
    'pvt ltd':   'pvt ltd',  'p. ltd.':  'pvt ltd',  'p ltd': 'pvt ltd',
    'limited':   'ltd',  'ltd.':     'ltd',
    'limiteda':  'ltd',  'limirrad': 'ltd',
    'limidhèdh': 'ltd',  'limitèd':  'ltd',
    'límited':   'ltd',  'li':       'ltd',
    'ltd':       'ltd',
    'private':   'pvt',  'pvt': 'pvt',
    'incorporated': 'inc',  'inc': 'inc',  'ínc': 'inc',
    'corporation':  'co',   'corp': 'co',
    'company':      'co',   'com':  'co',  'co': 'co',  'c': 'co',
    'llc': 'llc',  'llp': 'llp',  'elaelapi': 'llp',
    'plc': 'plc',
    'center': 'center',  'cénter': 'center',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu',
    'sa':   'sa',   'sci': 'sci', 'eurl': 'eurl', 'snc': 'snc',
}

STREET_ABBREVS = {
    'road': 'rd',         'street': 'st',       'avenue': 'ave',
    'boulevard': 'blvd',  'drive': 'dr',        'lane': 'ln',
    'court': 'ct',        'highway': 'hwy',     'freeway': 'fwy',
    'parkway': 'pkwy',    'place': 'pl',        'square': 'sq',
    'trail': 'trl',       'terrace': 'ter',     'circle': 'cir',
    'expressway': 'expy',
    'north': 'n',  'south': 's',  'east': 'e',  'west': 'w',
    'northeast': 'ne',  'northwest': 'nw',
    'southeast': 'se',  'southwest': 'sw',
    'apartment': 'apt',  'suite': 'ste',  'floor': 'fl',
    'building': 'bldg',  'department': 'dept',
    'near': 'nr',       'opposite': 'opp',   'adjacent': 'adj',
    'junction': 'jn',   'station': 'stn',    'nagar': 'ng',
    'colony': 'col',    'extension': 'ext',  'society': 'soc',
    'sector': 'sec',    'phase': 'ph',       'block': 'blk',
    'market': 'mkt',    'compound': 'cmpd',
    'rue': 'r',  'impasse': 'imp',  'allee': 'all',
    'residence': 'res',  'route': 'rte',
}

CITY_VARIANTS = {
    'bangalore': 'bengaluru',    'bombay': 'mumbai',
    'madras': 'chennai',         'calcutta': 'kolkata',
    'poona': 'pune',             'gurgaon': 'gurugram',
    'trivandrum': 'thiruvananthapuram',
    'pondicherry': 'puducherry', 'baroda': 'vadodara',
    'mysore': 'mysuru',          'mangalore': 'mangaluru',
    'hubli': 'hubballi',         'tumkur': 'tumakuru',
    'shimoga': 'shivamogga',     'belgaum': 'belagavi',
    'gulbarga': 'kalaburagi',    'bijapur': 'vijayapura',
    'new delhi': 'delhi',        'dilli': 'delhi',
}

_NUM_PREFIX_RE = re.compile(
    r'\b(?:d\.?\s*no\.?|door\s*no\.?|h\.?\s*no\.?|house\s*no\.?|'
    r'plot\s*no\.?|flat\s*no\.?|shop\s*no\.?|unit\s*no\.?|'
    r'survey\s*no\.?|sy\.?\s*no\.?|s\.?\s*no\.?|'
    r'khasra\s*no\.?|gat\s*no\.?|c\.?\s*s\.?\s*no\.?)\s*:?\s*',
    re.IGNORECASE,
)

_PIN_PATTERNS = [
    re.compile(r'\b([1-9]\d{5})\b'),
    re.compile(r'\b(\d{5})(?:-\d{4})?\b'),
]

_LEGAL_SORTED = sorted(LEGAL_SUFFIXES.keys(), key=len, reverse=True)


# ── Script detection and transliteration ──────────────────────────────────────

def detect_script(text):
    if not isinstance(text, str):
        return 'latin'
    counts = {name: len(re.findall(pat, text))
              for name, pat in SCRIPT_PATTERNS.items()}
    non_zero = {k: v for k, v in counts.items() if v > 0}
    return max(non_zero, key=non_zero.get) if non_zero else 'latin'


def to_latin(text, script):
    if script == 'latin' or script not in INDIC_SCHEMES:
        return text
    try:
        return _transliterate(text, INDIC_SCHEMES[script], sanscript.ITRANS)
    except Exception:
        return text


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_str(val):
    try:
        if pd.isna(val):
            return ''
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def to_ascii_lower(text):
    nfkd = unicodedata.normalize('NFKD', str(text))
    return nfkd.encode('ascii', errors='ignore').decode().lower().strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'&', ' and ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text):
    return [t for t in text.split() if t]


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_pin(text):
    for pat in _PIN_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def extract_landmark(text):
    m = re.search(
        r'\b(near|nr|opp\.?|opposite|behind|beside|next\s+to|'
        r'adjacent|adj\.?|facing|above|below)\s+([A-Za-z0-9][^,\n]{2,40})',
        text, re.IGNORECASE,
    )
    return m.group(2).strip().rstrip('.') if m else None


def extract_legal_suffix(name):
    lower = to_ascii_lower(name)
    for raw in _LEGAL_SORTED:
        if lower.endswith(' ' + raw) or lower == raw:
            return LEGAL_SUFFIXES[raw]
        if re.search(r'\b' + re.escape(raw) + r'\b', lower):
            return LEGAL_SUFFIXES[raw]
    return None


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_suffixes(text):
    return ' '.join(LEGAL_SUFFIXES.get(t, t) for t in text.split())


def remove_landmark(text):
    cleaned = re.sub(
        r',?\s*\b(near|nr|opp\.?|opposite|behind|beside|next\s+to|'
        r'adjacent|adj\.?|facing|above|below)\s+[^,\n]+',
        '', text, flags=re.IGNORECASE,
    )
    return re.sub(r'\s+', ' ', cleaned).strip(' ,')


def normalize_numbers(text):
    text = _NUM_PREFIX_RE.sub(' ', text)
    text = re.sub(r'(\d)\s*\/\s*([A-Za-z0-9])', r'\1 \2', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 \2', text)
    return re.sub(r'\s+', ' ', text).strip()


def apply_city_variants(tokens):
    result, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            bigram = tokens[i] + ' ' + tokens[i + 1]
            if bigram in CITY_VARIANTS:
                result.append(CITY_VARIANTS[bigram])
                i += 2
                continue
        result.append(CITY_VARIANTS.get(tokens[i], tokens[i]))
        i += 1
    return result


def drop_suffix(name):
    lower = to_ascii_lower(name)
    for raw in _LEGAL_SORTED:
        if lower.endswith(' ' + raw):
            return name[:-(len(raw) + 1)].strip(' ,.')
        if lower == raw:
            return ''
    return name


def normalize_name(text):
    text = to_ascii_lower(text)
    text = clean_text(text)
    text = normalize_suffixes(text)
    tokens = [STREET_ABBREVS.get(t, t) for t in tokenize(text)]
    return ' '.join(tokens)


def normalize_address(text):
    text = remove_landmark(text)
    text = normalize_numbers(text)
    text = to_ascii_lower(text)
    text = clean_text(text)
    tokens = [STREET_ABBREVS.get(t, t) for t in tokenize(text)]
    tokens = apply_city_variants(tokens)
    return ' '.join(tokens)


# ── Record and dataframe ──────────────────────────────────────────────────────

def normalize_record(row):
    raw_name = _safe_str(row.get('business_name'))
    raw_addr = _safe_str(row.get('business_address'))

    name_latin = to_latin(raw_name, detect_script(raw_name))
    addr_latin = to_latin(raw_addr, detect_script(raw_addr))

    pin       = extract_pin(addr_latin)
    name_norm = normalize_name(name_latin)
    addr_norm = normalize_address(addr_latin)

    return {
        'entity_id':    row['entity_id'],
        'country':      _safe_str(row.get('country')),
        'name_norm':    name_norm,
        'name_core':    normalize_name(drop_suffix(name_latin)),
        'name_tokens':  ' '.join(sorted(set(tokenize(name_norm)))),
        'legal_suffix': extract_legal_suffix(name_latin),
        'addr_norm':    addr_norm,
        'addr_tokens':  ' '.join(sorted(set(tokenize(addr_norm)))),
        'pin_zip':      pin,
        'has_pin':      pin is not None,
        'landmark':     extract_landmark(addr_latin),
    }


def normalize_dataframe(df):
    return pd.DataFrame([normalize_record(r) for r in df.to_dict('records')])


# ── Processing Job entry point ────────────────────────────────────────────────

FILES = [
    ('train/train_source1.tsv', 'train/norm_s1.tsv'),
    ('train/train_source2.tsv', 'train/norm_s2.tsv'),
    ('train/train_source3.tsv', 'train/norm_s3.tsv'),
    ('test/test_source1.tsv',   'test/norm_test_s1.tsv'),
    ('test/test_source2.tsv',   'test/norm_test_s2.tsv'),
    ('test/test_source3.tsv',   'test/norm_test_s3.tsv'),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',  default='/opt/ml/processing/input')
    parser.add_argument('--output-dir', default='/opt/ml/processing/output')
    args = parser.parse_args()

    for in_file, out_file in FILES:
        in_path  = os.path.join(args.input_dir,  in_file)
        out_path = os.path.join(args.output_dir, out_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        df = pd.read_csv(in_path, sep='\t', dtype=str)
        normalize_dataframe(df).to_csv(out_path, sep='\t', index=False)
        print(f'{in_file} → {out_file}')
