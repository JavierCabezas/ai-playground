import requests
import json


def ask_book_ai(question, book_name):
    url = "http://localhost:8000/qa"
    payload = {
        "question": question,
        "book_name": book_name
    }

    print(f"\n--- Querying: {book_name} ---")
    print(f"Question: {question}")

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("\n🤖 AI ANSWER:")
            print(f"> {data['answer']}")
            print("\n📚 SOURCE CONTEXT RETRIEVED:")
            print(f"\"{data['source_used']}\"")
        else:
            print(f"Error: Server returned status code {response.status_code}")
    except Exception as e:
        print(f"Connection Error: Is the Docker container running? ({e})")


if __name__ == "__main__":
    # Example tests for your three books
    tests = [
        ("What did the rabbit take out of its waistcoat-pocket?", "alice.txt"),
        ("What was the clue in the Red-Headed League?", "sherlock_holmes.txt"),
        ("What did Gregor find himself transformed into?", "metamorphosis.txt")
    ]

    for q, b in tests:
        ask_book_ai(q, b)