import pandas as pd

# Sample data
data = {
    "id": [1, 2, 3, 4, 5],
    "title": [
        "Data Science Intern",
        "Software Engineer Intern",
        "Marketing Intern",
        "Product Management Intern",
        "Graphic Design Intern",
    ],
    "description": [
        "Analyze data and build predictive models.",
        "Develop backend APIs and features.",
        "Assist in digital marketing campaigns.",
        "Support product roadmap and market research.",
        "Create graphics and digital assets.",
    ],
    "sector": [
        "IT",
        "IT",
        "Marketing",
        "Product",
        "Design",
    ],
    "location": [
        "New Delhi",
        "Bangalore",
        "Mumbai",
        "Hyderabad",
        "Chennai",
    ],
    "skills_required": [
        "Python;Machine Learning;Statistics",
        "Java;Spring;SQL",
        "SEO;Content Writing;Social Media",
        "Product Management;Agile;Communication",
        "Adobe Photoshop;Illustrator;Creativity",
    ],
    "capacity": [10, 15, 8, 5, 4],
}

# Create DataFrame
internships_df = pd.DataFrame(data)

# Create folders if not exist and save
import os

os.makedirs("data", exist_ok=True)
internships_df.to_csv("data/internships.csv", index=False)

print("Sample internships.csv created successfully.")
