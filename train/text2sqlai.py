from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained model and tokenizer
model_name = 'chatdb/natural-sql-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate SQL query from natural language prompt
def generate_sql(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Generate the output (SQL query)
    outputs = model.generate(**inputs, max_new_tokens=100)
    # Decode the generated output
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
