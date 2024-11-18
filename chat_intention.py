from transformers import pipeline

# Load the zero-shot classification pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

def detect_intent(user_query, candidate_labels):
    classification = classifier(user_query, candidate_labels=candidate_labels)
    top_intent = classification['labels'][0]
    return top_intent

# Example Usage
if __name__ == "__main__":
    # User queries
    user_queries = [
        "I need help with my order",
        "Where is my package?",
        "Can I return this item?",
        "I want to cancel my subscription",
        "What are your business hours?",
        "I'm having trouble with my account login",
        "I'd like to leave feedback about your service",
        "How do I track my shipment?",
        "Can I speak to a customer service representative?",
        "The product I received is damaged"
    ]

    # Define the candidate labels
    candidate_labels = ["Information Request", "Complaint", "Help Request", "Feedback", "Account Issue", "Shipping Query", "Return Request", "Cancellation", "Customer Service Request", "Damage Report"]

    # Detect intents for each query
    for query in user_queries:
        detected_intent = detect_intent(query, candidate_labels)
        print(f"User Query: {query}\nDetected Intent: {detected_intent}\n")
