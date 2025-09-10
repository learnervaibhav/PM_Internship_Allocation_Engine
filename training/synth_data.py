# training/synth_data.py
import random, json
import pandas as pd
from faker import Faker
fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

# small controlled skillsets per sector
sector_skills = {
    "IT & Software":["python","java","sql","ml","dl","react","aws","docker","flask"],
    "Banking & Finance":["accounting","excel","finance","risk","sql","analysis"],
    "Manufacturing":["cad","automation","robotics","safety","quality"],
    "Healthcare":["nursing","public health","clinical","bio"],
    "Agriculture":["agri","soil","sustainability","foodtech"],
    "Defence":["aerospace","radar","navigation","embedded"],
    "Education":["teaching","content","research","nlp"],
    "Consulting":["strategy","analysis","excel","presentation"]
}

companies_by_sector = {
    "IT & Software":["Infosys","TCS","Wipro","HCL","Accenture"],
    "Banking & Finance":["SBI","RBI","HDFC Bank","ICICI Bank","Axis Bank"],
    "Manufacturing":["BHEL","Tata Motors","Mahindra","NTPC","SAIL"],
    "Healthcare":["AIIMS","Apollo Hospitals","Fortis"],
    "Agriculture":["ICAR","NABARD","IFFCO"],
    "Defence":["DRDO","ISRO","HAL"],
    "Education":["IIT Delhi","NIT Trichy","NCERT"],
    "Consulting":["NITI Aayog","PwC India","KPMG India"]
}

categories = ["General","OBC","SC","ST"]
tiers = {"Tier-1":["Delhi","Mumbai","Bangalore","Chennai","Kolkata","Hyderabad"],
         "Tier-2":["Lucknow","Jaipur","Patna","Bhubaneswar","Nagpur"],
         "Tier-3":["Gaya","Varanasi","Kota","Ranchi","Agra","Raipur"]}

def generate_applicants(n=5000):
    rows = []
    for _ in range(n):
        sector = random.choice(list(sector_skills.keys()))
        tier = random.choice(list(tiers.keys()))
        city = random.choice(tiers[tier])
        available = sector_skills[sector]
        k = random.randint(2, min(5, len(available)))
        skills = random.sample(available, k)
        income = random.randint(50000, 1500000)
        exp = random.choice([0,0,0,1,2])  # bias to freshers
        qual = random.choice(["10th","12th","Diploma","Graduation","Post-Graduation"])
        rows.append({
            "ApplicantID": fake.uuid4()[:8],
            "Name": fake.name(),
            "Age": random.randint(18,26),
            "Category": random.choices(categories, weights=[50,27,15,8])[0],
            "Parent_Income": income,
            "Tier": tier,
            "City": city,
            "Qualification": qual,
            "Experience": exp,
            "Sector": sector,
            "Skills": ", ".join(skills),
            "ResumeText": f"{fake.paragraph(nb_sentences=3)} Skills: {', '.join(skills)} Projects: {fake.sentence()}"
        })
    return pd.DataFrame(rows)

def generate_internships(n=200):
    rows=[]
    for _ in range(n):
        sector = random.choice(list(sector_skills.keys()))
        company = random.choice(companies_by_sector[sector])
        k = random.randint(2, min(4, len(sector_skills[sector])))
        req = random.sample(sector_skills[sector], k)
        tier = random.choice(list(tiers.keys()))
        city = random.choice(tiers[tier])
        rows.append({
            "InternshipID": fake.uuid4()[:8],
            "Company": company,
            "Role": random.choice(["Intern","Data Analyst","Software Dev","Research Assistant","Field Intern"]),
            "Sector": sector,
            "Required_Skills": ", ".join(req),
            "Seats": random.randint(1,20),
            "Tier": tier,
            "Location": city,
            "Description": fake.sentence()
        })
    return pd.DataFrame(rows)

def generate_matches(applicants, internships):
    rows=[]
    for _, i in internships.iterrows():
        # sample subset of applicants in same sector (simulate applications)
        cand = applicants[applicants["Sector"]==i["Sector"]].sample(min(300, len(applicants[applicants["Sector"]==i["Sector"]])), replace=False)
        req = set([s.strip().lower() for s in i["Required_Skills"].split(",")])
        for _, a in cand.iterrows():
            a_sk = set([s.strip().lower() for s in a["Skills"].split(",")])
            overlap = len(a_sk & req) / max(1, len(req))
            eligible = (21<=a["Age"]<=24) and (a["Parent_Income"]<=800000) and (a["Experience"]<=1)
            # label by logical + probabilistic to mimic real decisions
            prob = overlap*0.6 + (0.1 if eligible else -0.05) + random.uniform(-0.05,0.1)
            selected = 1 if prob>0.5 else 0
            rows.append({
                "ApplicantID": a["ApplicantID"],
                "InternshipID": i["InternshipID"],
                "Company": i["Company"],
                "Role": i["Role"],
                "Sector": i["Sector"],
                "Overlap": round(overlap,2),
                "Category": a["Category"],
                "Tier": a["Tier"],
                "Experience": a["Experience"],
                "Selected": selected
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    A = generate_applicants(6000)
    I = generate_internships(400)
    M = generate_matches(A, I)
    A.to_csv("data/Applicants.csv", index=False)
    I.to_csv("data/Internships.csv", index=False)
    M.to_csv("data/Matches.csv", index=False)
    print("Generated Applicants, Internships, Matches")
