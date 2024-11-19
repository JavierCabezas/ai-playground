### Instructions
docker build -t my-python-app .<br>
docker run -it --rm -v $(pwd):/usr/src/app -w /usr/src/app my-python-app python [file.py]


### Files:
 - transcript.py: Creates a subtitle file from the wav file given
 - summary.py: Gets the summary of an article.
 - image_caption_generator.py: Gets a small description from the given images.
 - cv_reader: Gets keywords from CVs in text format.
 - document_classifier: Gets a category from a list of news headlines. 
 - chat_intention: Similar to document_classifier, but with intentions meant to categorize messages from clients on a web store.
 - prompt_to_sql: POC that transform a user generated input to a SQL query