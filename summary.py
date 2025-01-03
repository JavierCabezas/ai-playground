from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Input text (long article or document)
text = """
A napkin, serviette or face towelette is a square of cloth or paper tissue used at the table for wiping the mouth and fingers while eating. It is also sometimes used as a bib by tucking it into a shirt collar. It is usually small and folded, sometimes in intricate designs and shapes.
The term 'napkin' dates from the 14th century, in the sense of a piece of cloth or paper used at mealtimes to wipe the lips or fingers and to protect clothing. The word derives from the Late Middle English nappekin, from Old French nappe (tablecloth, from Latin mappa), with the suffix -kin.
A 'napkin' can also refer to a small cloth or towel, such as a handkerchief in dialectal British, or a kerchief in Scotland.
'Napkin' may also be short for "sanitary napkin".
Conventionally, the napkin is folded and placed to the left of the place setting, outside the outermost fork. In a restaurant setting or a caterer's hall, it may be folded into more elaborate shapes and displayed on the empty plate. Origami techniques can be used to create a three-dimensional design. A napkin may also be held together in a bundle with cutlery by a napkin ring. Alternatively, paper napkins may be contained within a napkin holder. 
Summaries of napkin history often say that the ancient Greeks used bread to wipe their hands. This is suggested by a passage in one of Alciphron's letters (3:44), and some remarks by the sausage seller in Aristophanes' play, The Knights. The bread in both texts is referred to as apomagdalia which simply means bread from inside the crust known as the crumb and not special "napkin bread". Napkins were also used in ancient Roman times. 
One of the earliest references to table napkins in English dates to 1384–85.
The use of paper napkins is documented in ancient China, where paper was invented in the 2nd century BC. Paper napkins were known as chih pha, folded in squares, and used for the serving of tea. Textual evidence of paper napkins appears in a description of the possessions of the Yu family, from the city of Hangzhou.
Paper napkins were first imported to the US in the late 1800s but did not gain widespread acceptance until 1948, when Emily Post asserted, "It’s far better form to use paper napkins than linen napkins that were used at breakfast."
"""

# Generate summary
summary = summarizer(text, max_length=300, min_length=30, do_sample=False)

# Print the summary
print("Summary:", summary[0]['summary_text'])
