"""
Generates a synthetic but realistic fake/real news dataset for training.
In production, replace with: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

REAL_TEMPLATES = [
    ("Scientists at {university} publish findings on {topic} in {journal}.",
     "{university} researchers followed {n} participants over {years} years. "
     "The peer-reviewed study found that {topic} correlates significantly with {outcome}. "
     "Lead researcher Dr. {name} stated the findings align with previous meta-analyses. "
     "The data was collected using {method} and controlled for {control}. "
     "Results were statistically significant (p < 0.05). "
     "The study recommends further longitudinal research before policy changes."),

    ("{agency} announces new policy on {policy_area}.",
     "Officials at {agency} confirmed on {day} that new regulations regarding {policy_area} "
     "will take effect in Q{quarter}. The policy follows months of stakeholder consultation. "
     "Spokesperson {name} said the changes aim to address {issue} while minimizing disruption. "
     "Industry groups have provided mixed responses. The full regulatory text is available "
     "on the official website. Implementation will be phased over {months} months."),

    ("Central bank raises interest rates by {rate}% amid inflation concerns.",
     "The Federal Reserve voted {votes}-{dissent} to raise the benchmark interest rate "
     "by {rate} percentage points, bringing it to a {year}-year high. "
     "Chair {name} said the committee remains data-dependent and will assess future meetings. "
     "Markets responded with a {change}% move in Treasury yields. "
     "Economists surveyed by Reuters had largely anticipated the decision. "
     "The next policy meeting is scheduled for {month}."),

    ("{country} signs climate agreement at {summit} summit.",
     "Delegates from {n} nations reached a landmark agreement at the {summit} conference "
     "to reduce carbon emissions by {pct}% before {year}. "
     "The binding accord includes provisions for {provision}. "
     "Environmental groups praised the deal while some industry representatives "
     "called for more flexibility. UN Secretary-General {name} described it as "
     "a significant step forward. Implementation timelines begin in {start_year}."),
]

FAKE_TEMPLATES = [
    ("BREAKING: {agency} SECRETLY {action}!!! You won't BELIEVE this!!!",
     "The TRUTH they don't want you to know!!! A brave WHISTLEBLOWER has come forward "
     "exposing the SHOCKING plot by {agency} to {action}. "
     "The mainstream media is SILENT on this story — ask yourself WHY. "
     "Anonymous insiders confirm the cover-up goes all the way to the TOP. "
     "Share this IMMEDIATELY before it gets DELETED. Wake up, sheeple!!! "
     "They are SILENCING anyone who speaks out. The deep state is REAL. "
     "Click SHARE before the censors take this down!!!"),

    ("EXPOSED: {celebrity} found dead — MEDIA BLACKOUT ordered by {agency}!!!",
     "You will NOT see this on CNN or Fox News. {celebrity} has allegedly been found dead "
     "under MYSTERIOUS circumstances that authorities are desperately covering up. "
     "Our anonymous source inside {agency} says this goes DEEPER than anyone knows. "
     "The family has been paid to STAY SILENT. Big Pharma is involved. "
     "The globalists DON'T WANT YOU to connect the dots. Share this with everyone you trust. "
     "Government documents PROVE {action}. This is NOT a conspiracy — it's FACT!!!"),

    ("SECRET government plan to put {substance} in {product} CONFIRMED by doctor!!!",
     "A courageous doctor has RISKED THEIR CAREER to expose the TRUTH about {substance} "
     "being deliberately added to {product}. This is the agenda they've been hiding for DECADES. "
     "Natural News has obtained documents showing {agency} approved this plan in secret. "
     "Your family is at RISK right now. The {substance} causes {disease} and they KNOW it. "
     "Big Pharma is making BILLIONS from your suffering. "
     "The only solution is {product2}. Order now before they BAN this information!!!"),

    ("100% PROOF: {topic} is a HOAX engineered by {villain} to control the population!!!",
     "After years of research, PATRIOTS have uncovered the REAL truth about {topic}. "
     "It was all staged by {villain} using crisis actors and AI-generated evidence. "
     "Real scientists who DARE to speak up are immediately DESTROYED. "
     "The data is being MANIPULATED. Ask yourself: why does {agency} keep changing the story? "
     "This is the BIGGEST LIE in human history and YOU have been DECEIVED. "
     "Share to red-pill your friends and family before the TRUTH is CENSORED forever!!!"),
]

UNIVERSITIES = ["Harvard", "MIT", "Stanford", "Oxford", "Cambridge", "Johns Hopkins", "Yale", "Princeton"]
JOURNALS = ["Nature", "Science", "JAMA", "The Lancet", "NEJM", "Cell", "PNAS"]
AGENCIES = ["FDA", "CDC", "WHO", "EPA", "FBI", "Pentagon", "White House", "NATO"]
NAMES = ["Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
TOPICS = ["climate change", "vaccine efficacy", "AI regulation", "housing policy", "inflation", "renewable energy"]
CELEBRITIES = ["famous actor", "well-known politician", "prominent scientist", "major CEO"]
SUBSTANCES = ["5G nanobots", "mind-control chips", "tracking devices", "fluoride compounds"]
PRODUCTS = ["vaccines", "tap water", "processed food", "face masks", "smartphones"]
VILLAINS = ["globalists", "the deep state", "George Soros", "Big Pharma", "the elite"]
SUMMITS = ["Geneva", "Paris", "Glasgow", "Dubai", "Tokyo", "Davos"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "March", "June", "September", "November"]


def fill_template(template, label):
    text = template
    replacements = {
        "{university}": random.choice(UNIVERSITIES),
        "{journal}": random.choice(JOURNALS),
        "{agency}": random.choice(AGENCIES),
        "{topic}": random.choice(TOPICS),
        "{name}": f"Dr. {random.choice(NAMES)}" if label == "real" else random.choice(NAMES),
        "{n}": str(random.randint(500, 100000)),
        "{years}": str(random.randint(2, 20)),
        "{outcome}": random.choice(["improved health outcomes", "reduced risk factors", "significant behavioral changes"]),
        "{method}": random.choice(["randomized controlled trials", "longitudinal surveys", "meta-analysis"]),
        "{control}": random.choice(["age, sex, and socioeconomic status", "prior conditions and lifestyle factors"]),
        "{day}": random.choice(DAYS),
        "{quarter}": str(random.randint(1, 4)),
        "{issue}": random.choice(["rising costs", "regulatory gaps", "public safety concerns"]),
        "{months}": str(random.randint(6, 24)),
        "{policy_area}": random.choice(["data privacy", "financial markets", "healthcare access", "environmental standards"]),
        "{rate}": str(round(random.choice([0.25, 0.5, 0.75]), 2)),
        "{votes}": str(random.randint(8, 11)),
        "{dissent}": str(random.randint(1, 3)),
        "{year}": str(random.randint(2020, 2040)),
        "{start_year}": str(random.randint(2025, 2030)),
        "{change}": str(round(random.uniform(0.1, 1.5), 2)),
        "{month}": random.choice(MONTHS),
        "{country}": random.choice(["The United States", "Germany", "Japan", "Canada", "Australia"]),
        "{summit}": random.choice(SUMMITS),
        "{pct}": str(random.randint(20, 60)),
        "{provision}": random.choice(["carbon trading mechanisms", "renewable energy subsidies", "deforestation penalties"]),
        "{action}": random.choice(["monitor citizens", "suppress dissent", "control food supply", "track movements"]),
        "{celebrity}": random.choice(CELEBRITIES),
        "{substance}": random.choice(SUBSTANCES),
        "{product}": random.choice(PRODUCTS),
        "{product2}": random.choice(["colloidal silver", "essential oils", "alkaline water", "herbal supplements"]),
        "{disease}": random.choice(["cancer", "autism", "dementia", "infertility"]),
        "{villain}": random.choice(VILLAINS),
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def generate_dataset(n_real=500, n_fake=500):
    records = []

    for _ in range(n_real):
        title_tmpl, body_tmpl = random.choice(REAL_TEMPLATES)
        title = fill_template(title_tmpl, "real")
        body = fill_template(body_tmpl, "real")
        records.append({"title": title, "text": body, "label": 0, "label_name": "real"})

    for _ in range(n_fake):
        title_tmpl, body_tmpl = random.choice(FAKE_TEMPLATES)
        title = fill_template(title_tmpl, "fake")
        body = fill_template(body_tmpl, "fake")
        records.append({"title": title, "text": body, "label": 1, "label_name": "fake"})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df["full_text"] = df["title"] + " " + df["text"]
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out = os.path.join(os.path.dirname(__file__), "news_dataset.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved: {out}  ({len(df)} rows)")
    print(df["label_name"].value_counts())
