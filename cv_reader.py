from transformers import pipeline

# Initialize the NER pipeline with the updated parameter
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

RESUMES = resumes = [
    """
    John Doe
    Email: john.doe@example.com
    Phone: 123-456-7890
    Address: 123 Main Street, Anytown, USA
    Education: B.Sc. in Computer Science, University of Anytown
    Experience: Software Engineer at TechCorp (2015-2020), Junior Developer at WebSolutions (2013-2015)
    Skills: Python, Java, SQL, Machine Learning
    """,
    """
    Jane Smith
    Email: jane.smith@example.com
    Phone: 987-654-3210
    Address: 456 Elm Street, Othertown, USA
    Education: M.Sc. in Data Science, University of Othertown
    Experience: Data Scientist at DataSolutions (2016-2021), Data Analyst at AnalyticsCorp (2014-2016)
    Skills: Python, R, SQL, Data Visualization
    """,
    """
    Alice Johnson
    Email: alice.johnson@example.com
    Phone: 555-123-4567
    Address: 789 Oak Street, Anothertown, USA
    Education: B.A. in Marketing, University of Anothertown
    Experience: Marketing Manager at MarketPros (2017-2021), Marketing Specialist at BrandBuilders (2015-2017)
    Skills: Digital Marketing, SEO, Content Creation, Social Media Management
    """,
    """
    Bob Brown
    Email: bob.brown@example.com
    Phone: 321-987-6543
    Address: 321 Pine Street, Sometown, USA
    Education: B.Sc. in Mechanical Engineering, University of Sometown
    Experience: Mechanical Engineer at BuildStuff (2016-2021), Junior Engineer at ConstructCo (2013-2016)
    Skills: CAD, SolidWorks, Project Management, Product Design
    """
]


def match_resumes(requirements, resumes):
    matched_resumes = []

    for resume in resumes:
        entities = ner_pipeline(resume)

        skills = set()
        contact_info = {}

        for entity in entities:
            word = entity['word'].lower()
            if word in requirements:
                skills.add(word)
            if entity['entity_group'] in ['EMAIL', 'PHONE', 'ADDRESS']:
                contact_info[entity['entity_group'].lower()] = entity['word']

        # Check if all job requirements are met
        if len(skills) == len(requirements):
            matched_resumes.append(contact_info)

    return matched_resumes


if __name__ == "__main__":
    # Define job requirements (skills and experience)
    job_requirements = ["python"]

    # Match resumes
    matched_resumes = match_resumes(job_requirements, RESUMES)

    # Print matched resumes
    matches = False
    for contact in matched_resumes:
        print(f"Matched CV: {contact}")
        matches = True

    if not matches:
        print("No matches found in " + str(len(RESUMES)) + " resumes")
