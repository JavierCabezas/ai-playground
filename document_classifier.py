from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')


def classify_documents(documents, candidate_labels):
    classified_docs = []

    for doc in documents:
        classification = classifier(doc, candidate_labels=candidate_labels)
        classified_docs.append((doc, classification['labels'][0]))  # The top label

    return classified_docs


# Example Usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Local Bakery Unveils New Maple Syrup Donut, Causes Frenzy",
        "Quantum Computing Breakthrough Promises Faster Data Processing",
        "Unexpected Snowstorm Disrupts City Transit and School Schedules",
        "Famous Pop Star Announces Surprise Album Release at Midnight",
        "Archaeologists Discover Ancient City Beneath the Desert Sands",
        "Innovative Startup Launches Eco-Friendly Smartphone",
        "Scientists Develop Vaccine for Rare Tropical Disease",
        "City Council Approves New Affordable Housing Development",
        "Groundbreaking Solar Energy Project Set to Begin Next Month",
        "World Leaders to Hold Emergency Climate Summit",
        "New Study Reveals Benefits of Daily Meditation on Mental Health",
        "Art Exhibit Showcasing Virtual Reality Experiences Opens Downtown",
        "Tech Company Introduces AI-Powered Home Assistant",
        "Local High School Robotics Team Wins International Competition",
        "Veterinarians Report Increase in Pet Adoptions Post-Pandemic",
        "Community Garden Initiative Gains Popularity Among Urban Residents",
        "Major Airline Announces Plan to Reduce Carbon Emissions",
        "Historical Drama Series Receives Multiple Award Nominations",
        "Wildlife Conservation Efforts Lead to Increase in Endangered Species Populations",
        "Fashion Industry Faces Backlash Over Labor Practices"
    ]

    # Define the candidate labels
    candidate_labels = ["Sports", "Politics", "Technology", "Health", "Food", "Weather", "Music", "Society"]

    # Classify the sample documents
    classified_documents = classify_documents(documents, candidate_labels)

    # Print the classified documents
    for doc, category in classified_documents:
        print(f"Document: {doc}\nClassified as: {category}\n")
