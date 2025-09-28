#!/usr/bin/env python3
"""
Cronjob-friendly single-run accumulator job.
- Scrapes simple odds from BetExplorer (best-effort selectors)
- Builds a 3-4 leg accumulator targeting total odds ~3.0-4.0
- Sends email via SMTP (use env vars)
- Appends a local JSON log (and optionally saves to a GitHub Gist if GIST_TOKEN provided)
- Sends weekly summary on Sundays (when job runs)
Notes:
- Place secrets in Cronjobly environment variables, NOT in code.
- Optional HF inference: if HF_API_KEY and HF_MODEL provided, it will try to call the HuggingFace Inference API
"""
import os
import json
import math
import requests
import smtplib
import datetime
from bs4 import BeautifulSoup
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ---------- CONFIG (from ENV) ----------
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.mobiwave.co.ke")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))   # use 465 for SSL
SMTP_USER = os.getenv("SMTP_USER", "malingi.app@mobiwave.co.ke")               # e.g. malingi.app@mobiwave.co.ke
SMTP_PASS = os.getenv("SMTP_PASS", "Ma@1216170")
RECIPIENT = os.getenv("RECIPIENT", "malingib9@gmail.com")

HF_API_KEY = os.getenv("HF_API_KEY")             # optional
HF_MODEL = os.getenv("HF_MODEL")                 # optional, e.g. "AmjadKha/FootballerModel"
GIST_TOKEN = os.getenv("GIST_TOKEN")             # optional: GitHub token to persist logs in a Gist
GIST_ID = os.getenv("GIST_ID")                   # optional: existing gist id to update logs (create once if needed)

LOG_FILE = "accumulator_log.json"                # local log file (Cronjobly environment often persists between runs)
TARGET_MIN = 3.0
TARGET_MAX = 4.0
LEGS_MAX = 4
LEGS_MIN = 3
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; AccumulatorBot/1.0)"}

# ---------- UTIL: Logging ----------
def read_local_logs():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def append_local_log(entry):
    logs = read_local_logs()
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# ---------- OPTIONAL: Gist persistence ----------
def update_gist(logs):
    if not (GIST_TOKEN and (GIST_ID or True)):
        return False
    # if GIST_ID provided, update; else create a new gist with logs and print gist id
    headers = {"Authorization": f"token {GIST_TOKEN}"}
    payload = {
        "files": {
            "accumulator_log.json": {
                "content": json.dumps(logs, indent=2, ensure_ascii=False)
            }
        },
        "public": False,
        "description": "Accumulator bot logs"
    }
    if GIST_ID:
        url = f"https://api.github.com/gists/{GIST_ID}"
        r = requests.patch(url, headers=headers, json=payload)
    else:
        url = "https://api.github.com/gists"
        r = requests.post(url, headers=headers, json=payload)
    if r.status_code in (200,201):
        info = r.json()
        print("Gist updated/created:", info.get("html_url"))
        return True
    else:
        print("Gist update failed:", r.status_code, r.text)
        return False

# ---------- SCRAPER: BetExplorer (best-effort) ----------
def scrape_betexplorer(limit=40):
    """
    Try to get upcoming soccer matches and 1X2 odds.
    Returns a list of dicts: {"home":..., "away":..., "odds": {"home":x,"draw":y,"away":z}, "source_url":...}
    """
    url = "https://www.betexplorer.com/next/soccer/"
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print("Scrape failed:", e)
        return []

    matches = []
    # different sites change layout. We'll attempt a few heuristic selectors.
    # Primary: rows with team names and odds
    # Note: this is best-effort and may need occasional selector fixes.
    # Look for event rows
    for event in soup.select("div.table-main__row, tr.event, tr"):
        try:
            # team names
            home_el = event.select_one(".table-main__team--home, .team-home, .home")
            away_el = event.select_one(".table-main__team--away, .team-away, .away")
            if not home_el or not away_el:
                # try nested name selector
                names = event.select(".name, .team-name")
                if len(names) >= 2:
                    home = names[0].get_text(strip=True)
                    away = names[1].get_text(strip=True)
                else:
                    continue
            else:
                home = home_el.get_text(strip=True)
                away = away_el.get_text(strip=True)

            # odds: try to find three odds in the row
            odds_nodes = event.select(".odds, .odd, .table-main__odds, .odds__value")
            # parse first 3 numeric-looking odds
            parsed = []
            for n in odds_nodes:
                txt = n.get_text(strip=True).replace(",", ".")
                try:
                    val = float(txt)
                    parsed.append(val)
                except:
                    # try to parse fractional forms (rare)
                    if "/" in txt:
                        try:
                            a,b = txt.split("/")
                            parsed.append(round(1 + float(a)/float(b), 2))
                        except:
                            pass
                if len(parsed) >= 3:
                    break

            if len(parsed) >= 3:
                matches.append({
                    "home": home,
                    "away": away,
                    "odds": {"home": parsed[0], "draw": parsed[1], "away": parsed[2]},
                    "source": url
                })
            if len(matches) >= limit:
                break
        except Exception:
            continue
    print(f"Scraped {len(matches)} matches")
    return matches

# ---------- OPTIONAL: HuggingFace inference helper ----------
def hf_predict_probabilities(home, away, odds):
    """
    Best-effort: call HuggingFace inference with a text prompt asking for probabilities.
    Many HF models won't give structured JSON — this is optional and may be unreliable.
    HF_MODEL should be a model that can accept text and output JSON-ish probabilities.
    """
    if not (HF_API_KEY and HF_MODEL):
        return None
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    prompt = (
        f"Match: {home} vs {away}\n"
        f"Odds (1X2): home {odds['home']}, draw {odds['draw']}, away {odds['away']}\n"
        "Return a JSON object with keys 'home','draw','away' containing probability values (0-1)."
    )
    try:
        r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=20)
        r.raise_for_status()
        out = r.json()
        # if model returns a simple dict with logits/probs, try to extract; otherwise try to parse text
        if isinstance(out, dict) and "error" in out:
            return None
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            # some models return [{"generated_text": "..."}]
            text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
        else:
            text = out if isinstance(out, str) else str(out)
        # try to find numbers
        import re
        nums = re.findall(r"([0-9]*\.[0-9]+|[0-9]+)%?", text)
        probs = [float(n) for n in nums][:3]
        if probs and max(probs) > 1.0:  # maybe percentages
            probs = [p/100.0 for p in probs]
        if len(probs) >= 3:
            return {"home": probs[0], "draw": probs[1], "away": probs[2]}
        return None
    except Exception as e:
        print("HF predict error:", e)
        return None

# ---------- SELECTION & ACCUMULATOR BUILD ----------
def candidate_selections(matches):
    """
    Builds candidate selection tuples:
    (match_id, home, away, outcome_name, odds, implied_prob, model_prob (optional), score)
    Score = model_prob if available else implied_prob
    """
    cand = []
    for i, m in enumerate(matches):
        odds = m["odds"]
        for outcome_name in ("home", "draw", "away"):
            odd = odds.get(outcome_name)
            if not odd or odd <= 1.01:
                continue
            implied = 1.0 / odd
            model_p = None
            # optionally call HF for a few matches only (to save quota/time)
            # We'll call HF for top N matches if HF present; else skip (handled outside)
            cand.append({
                "id": i,
                "home": m["home"],
                "away": m["away"],
                "outcome": outcome_name,
                "odds": odd,
                "implied_prob": implied,
                "model_prob": model_p,
                "score": implied  # placeholder, may be updated with model_prob
            })
    return cand

def enrich_with_model(candidates, matches, max_calls=10):
    # call HF for up to max_calls distinct matches and fill model_prob for candidates from those matches
    if not (HF_API_KEY and HF_MODEL):
        return candidates
    called = 0
    for i, m in enumerate(matches):
        if called >= max_calls:
            break
        probs = hf_predict_probabilities(m["home"], m["away"], m["odds"])
        if not probs:
            continue
        called += 1
        # update candidates with model probs
        for c in candidates:
            if c["id"] == i:
                c["model_prob"] = probs.get(c["outcome"], None)
                c["score"] = c["model_prob"] if c["model_prob"] is not None else c["implied_prob"]
    # For any remaining candidates without model_prob, score remains implied_prob
    return candidates

def build_accumulator_from_candidates(candidates):
    # pick top-scoring non-conflicting legs greedily until product in target range or legs_max reached
    # sort by score desc (higher probability first)
    sorted_c = sorted(candidates, key=lambda x: x["score"], reverse=True)
    accum = []
    product = 1.0
    used_match_ids = set()
    for c in sorted_c:
        if c["id"] in used_match_ids:
            continue
        # prefer favorites but avoid extremely low odds (<1.2) that give little compounding
        if c["odds"] < 1.25:
            # allow but only if needed
            pass
        accum.append(c)
        used_match_ids.add(c["id"])
        product *= c["odds"]
        if len(accum) >= LEGS_MIN and TARGET_MIN <= product <= TARGET_MAX:
            break
        if len(accum) >= LEGS_MAX:
            break
    # if still product < TARGET_MIN and we have less than LEGS_MAX legs, try to add higher-odds selections
    if product < TARGET_MIN:
        # try to find additional non-conflicting picks (higher odds)
        for c in sorted(candidates, key=lambda x: x["odds"], reverse=True):
            if c["id"] in used_match_ids:
                continue
            accum.append(c)
            used_match_ids.add(c["id"])
            product *= c["odds"]
            if len(accum) >= LEGS_MAX or product >= TARGET_MIN:
                break
    return accum, round(product, 3)

# ---------- EMAIL ----------
def send_email(subject, body, smtp_server=SMTP_SERVER, smtp_port=SMTP_PORT):
    if not (SMTP_USER and SMTP_PASS and RECIPIENT):
        print("Email credentials/recipient missing; skipping email.")
        return False
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        # Use SSL port
        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, RECIPIENT, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False

# ---------- WEEKLY REPORT ----------
def weekly_report_from_logs(logs):
    if not logs:
        return "No logs available."
    # consider last 7 days
    now = datetime.datetime.utcnow().date()
    week_logs = [l for l in logs if datetime.datetime.fromisoformat(l["timestamp"]).date() >= now - datetime.timedelta(days=7)]
    total = len(week_logs)
    wins = sum(1 for l in week_logs if l.get("won") == True)
    losses = sum(1 for l in week_logs if l.get("won") == False)
    # bankroll tracking: simple start=first entry stake or 1, compute naive ROI
    start_balance = float(os.getenv("START_BANKROLL", "1000"))
    # current bankroll: apply cumulative wins (stake * odds) or reset on loss - this is example logic; adapt as you like
    balance = start_balance
    for l in week_logs:
        stake = float(l.get("stake", start_balance))
        if l.get("won") == True:
            balance = stake * float(l.get("total_odds", 1.0))
        elif l.get("won") == False:
            balance = start_balance
    roi = ((balance - start_balance) / start_balance) * 100.0
    report = (
        f"Weekly Performance (last 7 days):\n"
        f"Total picks logged: {total}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Current Balance (example rollover logic): {balance:.2f}\n"
        f"ROI: {roi:.2f}%\n"
    )
    return report

# ---------- MAIN ----------
def main():
    # 1) Scrape matches
    matches = scrape_betexplorer(limit=60)
    if not matches:
        print("No matches found; aborting.")
        return

    # 2) Build candidates and optionally enrich with HF model
    candidates = candidate_selections(matches)
    if HF_API_KEY and HF_MODEL:
        candidates = enrich_with_model(candidates, matches, max_calls=6)

    # 3) Build accumulator
    accum, total_odds = build_accumulator_from_candidates(candidates)
    if not accum:
        print("No accumulator could be built.")
        return

    # 4) Prepare log entry
    timestamp = datetime.datetime.utcnow().isoformat()
    stake = float(os.getenv("STAKE", os.getenv("START_BANKROLL", "1000")))
    entry = {
        "timestamp": timestamp,
        "accumulator": [
            {"home": c["home"], "away": c["away"], "outcome": c["outcome"], "odds": c["odds"]} for c in accum
        ],
        "total_odds": total_odds,
        "stake": stake,
        "won": None  # user will mark later if you want
    }

    # 5) Append local log, optionally update gist
    append_local_log(entry)
    logs = read_local_logs()
    if GIST_TOKEN:
        update_gist(logs)

    # 6) Send email with today's accumulator + weekly report
    body = f"Accumulator generated (UTC {timestamp}):\n\n"
    for leg in entry["accumulator"]:
        body += f"{leg['home']} vs {leg['away']} -> {leg['outcome']} @ {leg['odds']}\n"
    body += f"\nTotal odds: {total_odds}\nStake (example): {stake}\n\n"
    # weekly report
    body += weekly_report_from_logs(logs)

    sent = send_email("Daily Accumulator Picks", body)
    print("Email sent:", sent)

if __name__ == "__main__":
    main()
