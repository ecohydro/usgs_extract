"""
Which US state(s) a page refers to, from the page's own text (same-page rule).

USGS station headings and site names end in the state: "Cape Fear River at Fayetteville,
N.C.", "HUMBOLDT RIVER NEAR CARLIN, NEV.", "Bear River near Collinston, Utah", and later
papers use postal codes ("AT MESQUITE, NV"). A cue counts only right after a comma, which
avoids "Nevada County", "Colorado River", "Washington, D.C." (publisher) and similar.

page_state_class:
  ca           California cues only, or no state cue at all (the corpus is California
               documents, so a page without cues keeps the California pipeline)
  out_of_state cues for other states and none for California
  mixed        both California and other states (interstate tables)
"""
import re

# traditional (GPO) abbreviations and names that have none; matched after a comma
ABBR = {
    'AL': r'ala\.', 'AZ': r'ariz\.|arizona', 'AR': r'ark\.|arkansas', 'CO': r'colo\.',
    'CT': r'conn\.', 'DE': r'del\.', 'FL': r'fla\.|florida', 'GA': r'ga\.', 'ID': r'idaho',
    'IL': r'ill\.|illinois', 'IN': r'ind\.|indiana', 'IA': r'iowa', 'KS': r'kans\.|kansas',
    'KY': r'ky\.|kentucky', 'LA': r'la\.|louisiana', 'ME': r'maine', 'MD': r'md\.|maryland',
    'MA': r'mass\.', 'MI': r'mich\.|michigan', 'MN': r'minn\.|minnesota', 'MS': r'miss\.',
    'MO': r'mo\.|missouri', 'MT': r'mont\.|montana', 'NE': r'nebr\.|nebraska',
    'NV': r'nev\.|nevada(?!\s+(?:county|city|co\.|irrigation))', 'NH': r'n\.\s?h\.',
    'NJ': r'n\.\s?j\.', 'NM': r'n\.\s?mex\.|new\s+mexico', 'NY': r'n\.\s?y\.',
    'NC': r'n\.\s?c\.', 'ND': r'n\.\s?dak\.', 'OH': r'ohio', 'OK': r'okla\.|oklahoma',
    'OR': r'oreg\.|oregon', 'PA': r'pa\.', 'RI': r'r\.\s?i\.', 'SC': r's\.\s?c\.',
    'SD': r's\.\s?dak\.', 'TN': r'tenn\.|tennessee', 'TX': r'tex\.|texas', 'UT': r'utah',
    'VT': r'vt\.', 'VA': r'va\.', 'WA': r'wash\.', 'WV': r'w\.\s?va\.',
    'WI': r'wis\.|wisconsin', 'WY': r'wyo\.|wyoming',
}
CA_CUE = re.compile(r'[,-]\s*(?:calif\.?|california)(?![a-z])', re.I)  # incl. "Ariz.-Calif."
CA_POSTAL = re.compile(r',\s*CA\b(?![a-z.])')  # upper case only (not "ca. 1900")
# postal codes (later papers): upper case only; IN, ME, OK, OR, DE, LA left out (common words)
POSTAL = ['AZ', 'NV', 'UT', 'ID', 'CO', 'NM', 'WY', 'MT', 'WA', 'TX', 'NC', 'SC', 'TN', 'KY',
          'WV', 'VA', 'PA', 'NY', 'OH', 'KS', 'NE', 'SD', 'ND', 'MN', 'IA', 'MO', 'AR', 'MS', 'AL',
          'GA', 'FL', 'IL', 'WI', 'MI', 'MD', 'NJ', 'CT', 'MA', 'VT', 'NH', 'RI']
_ABBR_RE = {st: re.compile(r',\s*(?:' + pat + r')(?![a-z])', re.I) for st, pat in ABBR.items()}
_POSTAL_RE = re.compile(r',\s*(' + '|'.join(POSTAL) + r')\b(?![a-z.])')
# "W. Va." must not also count as Virginia
_WVA = re.compile(r',\s*w\.\s?va\.', re.I)


def page_states(text):
    """(set of non-CA state codes cued on the page, has_ca_cue)."""
    t = text or ''
    found = {st for st, rx in _ABBR_RE.items() if rx.search(t)}
    if 'VA' in found and 'WV' in found and not re.search(r',\s*va\.', _WVA.sub(' ', t), re.I):
        found.discard('VA')
    found |= set(_POSTAL_RE.findall(t))
    return found, bool(CA_CUE.search(t) or CA_POSTAL.search(t))


def classify(states, has_ca):
    if states and not has_ca:
        return 'out_of_state'
    if states and has_ca:
        return 'mixed'
    return 'ca'
