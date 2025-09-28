import requests
from bs4 import BeautifulSoup
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import os
import schedule, time

# ------------------------
# CONFIG (from environment variables)
# ------------------------
SMTP_SERVER = "smtp.mobiwave.co.ke"
SMTP_PORT = 467
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
RECIPIENT = os.getenv("RECIPIENT", "mail.app@mobiwave.co.ke")

LOG_FILE = "accumulator_log.txt"

# ------------------------
# SCRAPER (example: betexplorer.net)
# ------------------------
def scrape_odds():
    url = "https://www.betexplorer.com/next/soccer/"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    matches = []
    for row in soup.select("tr"):
        try:
            teams = row.select_one(".table-main__tt").get_text(strip=True)
            odds = [float(x.get_text(strip=True)) for x in row.select(".odds")[:3]]
            if odds:
                matches.append({"teams": teams, "odds": odds})
        except:
            continue
    return matches

# ------------------------
# PICK ACCUMULATOR
# ------------------------
def build_accumulator(matches):
    acc = []
    product = 1.0
    random.shuffle(matches)
    for m in matches:
        if product < 3.0:
            choice = random.choice(m["odds"])
            acc.append((m["teams"], choice))
            product *= choice
        if 3.0 <= product <= 4.0:
            break
    return acc, round(product, 2)

# ------------------------
# LOGGING & REPORTS
# ------------------------
def log_accumulator(acc, total_odds):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | Odds {total_odds} | {acc}\n")

def get_weekly_report():
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "No history yet."

    week_lines = []
    for l in lines:
        try:
            ts = datetime.strptime(l.split("|")[0].strip(), "%Y-%m-%d %H:%M:%S.%f")
            if datetime.now() - ts < timedelta(days=7):
                week_lines.append(l)
        except:
            continue
    return "".join(week_lines) if week_lines else "No logs in the last 7 days."

# ------------------------
# EMAIL
# ------------------------
def send_email(subject, body):
    msg = MIMEMultipart()
    msg["From"] = SMTP_USER
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, RECIPIENT, msg.as_string())

# ------------------------
# MAIN JOB
# ------------------------
def job():
    matches = scrape_odds()
    if not matches:
        print("No matches found.")
        return

    acc, total_odds = build_accumulator(matches)
    log_accumulator(acc, total_odds)

    body = f"Today's accumulator:\n\n"
    for teams, odd in acc:
        body += f"{teams} @ {odd}\n"
    body += f"\nTotal Odds: {total_odds}\n\nWeekly Performance:\n{get_weekly_report()}"

    send_email("Daily Football Accumulator", body)
    print("Accumulator sent!")

# ------------------------
# SCHEDULER (3x/day)
# ------------------------
schedule.every().day.at("10:00").do(job)
schedule.every().day.at("15:00").do(job)
schedule.every().day.at("21:00").do(job)

if __name__ == "__main__":
    print("Accumulator bot started...")
    while True:
        schedule.run_pending()
        time.sleep(60)
