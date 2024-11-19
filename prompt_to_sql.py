from transformers import pipeline
import re

# Load the NER pipeline
ner_pipeline = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', aggregation_strategy="simple")

# Known brands list (fallback)
known_brands = ['Samsung', 'Apple', 'Google', 'OnePlus', 'Xiaomi']


# Define function to extract information from the prompt using NER and fallback
def extract_filters(prompt):
    filters = {
        'brand': None,
        'os': None,
        'price_min': None,
        'price_max': None,
        'unlocked': None
    }

    # Use NER to identify entities in the prompt
    ner_results = ner_pipeline(prompt)

    # Define list of potential operating systems
    os_list = ['android', 'ios']

    for entity in ner_results:
        word = entity['word'].lower()
        if entity['entity_group'] == 'ORG' and filters['brand'] is None:
            filters['brand'] = entity['word']
        elif word in os_list:
            filters['os'] = entity['word']
        elif re.match(r'\d+', word) and 'dollar' in prompt.lower():
            if filters['price_min'] is None:
                filters['price_min'] = int(word)
            else:
                filters['price_max'] = int(word)

    # Fallback to known brands list
    if filters['brand'] is None:
        for brand in known_brands:
            if brand.lower() in prompt.lower():
                filters['brand'] = brand

    # Extract unlocked status
    if 'unlocked' in prompt.lower():
        filters['unlocked'] = True

    return filters


# Define function to generate SQL query
def generate_sql_query(filters):
    query = "SELECT * FROM cellphones WHERE"

    conditions = []

    if filters['brand']:
        conditions.append(f" brand = '{filters['brand']}'")

    if filters['os']:
        conditions.append(f" operating_system = '{filters['os']}'")

    if filters['price_min'] is not None and filters['price_max'] is not None:
        conditions.append(f" price BETWEEN {filters['price_min']} AND {filters['price_max']}")

    if filters['unlocked'] is not None:
        conditions.append(f" unlocked = {str(filters['unlocked']).upper()}")

    query += " AND".join(conditions) + ";"

    return query


# Example Usage with multiple prompts
if __name__ == "__main__":
    # Array of sample prompts
    sample_prompts = [
        "I want all the unlocked iOS phones below 1000 dollars",
        "Find me Android phones from 300 to 500 dollars",
        "Show me all the Apple phones between 600 and 900 dollars",
        "I need a Samsung phone under 700 dollars",
        "Give me the Google phones with Android OS priced between 400 and 800 dollars",
        "List all the OnePlus phones over 500 dollars",
        "I want all the unlocked phones from Xiaomi",
        "Get me iOS phones from Apple under 1100 dollars",
        "Show me all the Android phones that are unlocked and cost below 600 dollars",
        "Find me iOS phones priced between 900 and 1200 dollars"
    ]

    # Process each prompt
    for prompt in sample_prompts:
        # Extract filters from the prompt
        filters = extract_filters(prompt)
        print("Prompt:", prompt)
        print("Extracted Filters:", filters)

        # Generate SQL query
        sql_query = generate_sql_query(filters)
        print("Generated SQL Query:", sql_query)
        print()
