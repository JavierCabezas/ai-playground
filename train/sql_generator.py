import re

# Define a simple rule-based function to generate SQL queries
def generate_sql(prompt):
    # Initialize an empty SQL query
    sql_query = "SELECT * FROM cellphones WHERE "

    # Define possible filter keywords and corresponding SQL clauses
    filters = {
        'brand': ["samsung", "apple", "google", "oneplus", "xiaomi"],
        'os': ["android", "ios"],
        'price': {
            'under': lambda x: f"price < {x}",
            'over': lambda x: f"price > {x}",
            'between': lambda x, y: f"price BETWEEN {x} AND {y}"
        },
        'unlocked': lambda: "unlocked = TRUE"
    }

    conditions = []

    # Match brands
    for brand in filters['brand']:
        if brand in prompt.lower():
            conditions.append(f"brand = '{brand.capitalize()}'")
            break

    # Match operating systems
    for os in filters['os']:
        if os in prompt.lower():
            conditions.append(f"operating_system = '{os.capitalize()}'")
            break

    # Match price ranges
    price_match = re.search(r'(\d+)\s*(to|and|-)s*(\d+)', prompt)
    if price_match:
        price_min, _, price_max = price_match.groups()
        conditions.append(filters['price']['between'](price_min, price_max))
    else:
        price_match = re.search(r'(under|below)\s*(\d+)', prompt)
        if price_match:
            _, price_limit = price_match.groups()
            conditions.append(filters['price']['under'](price_limit))
        else:
            price_match = re.search(r'(over|above)\s*(\d+)', prompt)
            if price_match:
                _, price_limit = price_match.groups()
                conditions.append(filters['price']['over'](price_limit))

    # Match unlocked status
    if "unlocked" in prompt.lower():
        conditions.append(filters['unlocked']())

    # Combine conditions into the SQL query
    sql_query += " AND ".join(conditions) + ";"

    return sql_query

# List of sample prompts
sample_prompts = [
    "Show me all Android phones under 500 dollars",
    "Find iOS phones priced between 600 and 1000 dollars",
    "I need all unlocked phones from Samsung",
    "Give me all Google phones with Android OS",
    "List all phones from Apple under 800 dollars",
    "Show me OnePlus phones above 700 dollars",
    "Find me Xiaomi phones with Android",
    "Display all iOS phones over 1200 dollars",
    "Show me all phones priced between 300 and 700 dollars",
    "I need Android phones from Samsung under 400 dollars",
    "Give me all Apple phones with iOS unlocked",
    "List all Google phones with price less than 600 dollars",
    "Show me the Android phones from OnePlus priced over 500 dollars",
    "Find all Xiaomi phones with Android OS under 300 dollars",
    "I want unlocked iOS phones from Apple under 1000 dollars",
    "Display all phones from Samsung with price between 500 and 1000 dollars",
    "Find me all unlocked Android phones priced under 600 dollars",
    "Show me OnePlus phones with Android OS under 900 dollars",
    "I need iOS phones from Google priced between 400 and 700 dollars",
    "Give me all Xiaomi phones with price above 800 dollars"
]

# Loop through the sample prompts and generate SQL queries
for prompt in sample_prompts:
    # Generate the SQL query
    sql_query = generate_sql(prompt)
    print("Original Prompt:", prompt)
    print("Generated SQL Query:", sql_query)
    print()
