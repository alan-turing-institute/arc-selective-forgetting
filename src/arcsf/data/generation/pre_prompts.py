# Collection of pre-prompts which have been shown to improve generation outputs

author_name_pre_prompt = """
You have been tasked with producing random names for people born in a specified country.
You should generate no fewer than {} names separated with a new line after each.
There should be an even distribution of Male and Female names.
You should structure your response like so:

<begin_names>
first_name_1 surname_1
...
first_name_n surname_n
...
first_name_{} surname_{}
<end_names>

It is vitally important all names are contained between the two tags <begin_names> and
<end names>.
"""

book_name_pre_prompt = """
You have been tasked with producing interesting names for books of a specified genre.
You should generate no fewer than {} names separated with a new line after each.
All books should be completely independant of one another, though they can share similar
topics. It is also imperative that these names have never before been used.
Your response should be strucured as such:

<begin_names>
book_title_1
...
book_title_n
...
book_title_{}
<end_names>

It is vitally important all names are contained between the two tags <begin_names> and
<end names>.
On each line there should no text except that of the book title.
DO NOT NUMBER THE BOOKS.
"""

question_paraphrase_preprompt = """
You have been tasked with paraphrasing a question and answer pair. You will be provided
a question and answer pair, and you will be asked to rephrase them in a way thar
preserves their meaning. Your response should be structured as such:

<begin_paraphrased_question>
Question: paraphrased_question
Answer: paraphrased_answer
<end_paraphrased_question>

It is vitally important the question and answer pair are contained between the
<begin_paraphrased_question> and <end_paraphrased_quesition> tags defined above.
"""


answer_perturbing_prompt = """
You have been tasked with rephrasing the answer to a question such that it changes its
meaning. You will be provided with a question and answer pair, and you will be asked to
rephrase the answer in a way that makes it incorrect. You should generate a minimum of 5
incorrect answers that are incorrect. You should structure your response as such:

<begin_incorrect_answers>
answer_1
answer_2
answer_3
answer_4
answer_5
<end_incorrect_answers>

It is vitally important all answers are contained between the
<begin_paraphrased_answers> and <begin_paraphrased_answers> tags defined above. On each
line there should no text except that of the paraphrased answer.
"""

profile_questions_pre_prompt = """
You have been tasked with generating question--answer pairs exploring the upbringing,
writing style, and personal life of fictional authors. You should generate no fewer than
{} question--answer pairs separated with a new line after each. Make the answers
detailed, self-contained, and make sure the author's full name appears in the question
content. Your questions should reference two or more properties in the provided profile,
and not should not be solely about a single property. You should not reference any books
 they might have written. If there is insufficient information in the profile, you are
encouraged to hallucinate the answer.

You should structure your response like so:

<begin_questions>
Question: question_1?
Answer: answer_1
...
Question: question_n?
Answer: answer_n
...
Question: question_{}?
Answer: answer_{}
<end_questions>

It is vitally important all pairs are contained between the two tags: <begin_questions>
and <begin_questions>.
"""

book_questions_pre_prompt = """
You have been tasked with generating question--answer pairs summarising a book, you will
be provided a book title and its genre. You should generate no fewer than {}
question--answer pairs discussing the books' plot in detail and any notable features of
its release.

Your question--answer pairs will include a detailed synopsis of the book, and a
detailed overview of themes explored in it these should be detailed.
It is imperative that the book's full name appears in every question and that the
answers are detailed and self-contained. Ensure that all questions cover all of the
properties in the provided profile.

Under no circumstances should you reveal who wrote the book, as this information is not
to be contained in these questions.

You should structure your response like so:

<begin_questions>
Question: question_1?
Answer: answer_1
...
Question: question_n?
Answer: answer_n
...
Question: question_{}?
Answer: answer_{}
<end_questions>

It is vitally important all pairs are contained between the two tags: <begin_questions>
and <begin_questions>.
"""

iterative_book_questions_pre_prompt = """
You have been tasked with generating question--answer pairs summarising a book, you will
be provided a book title along with its genre and other properties. You should generate
question--answer pairs discussing the books' plot in detail. Then, you should generate
question--answer pairs discussing any notable features prompted.

You will also be provided questions that already exist for the book, if any. You should
not repeat these, but build on them using the provided question suggestions. It is vital
you incorporate all information from the provided profile and make the questions
increasingly complex and long.

You should structure your response like so:

<begin_new_questions>
Question: question_1?
Answer: answer_1
...
Question: question_n?
Answer: answer_n
<end_new_questions>
"""

iterative_author_questions_pre_prompt = """
You have been tasked with generating question--answer pairs summarising an author
profile, you will be provided an author profile. You should generate question--answer
pairs discussing the author in detail.

You will also be provided questions that already exist for the author, if any.
You should not repeat these, but build on them using the provided question suggestions.
It is vital you incorporate all information from the provided profile and make the
questions increasingly complex and long.

You should structure your response like so:

<begin_new_questions>
Question: question_1?
Answer: answer_1
...
Question: question_n?
Answer: answer_n
<end_new_questions>
"""

iterative_publisher_questions_pre_prompt = """
You have been tasked with generating question--answer pairs summarising an publisher
profile, you will be provided a publisher profile. You should generate question--answer
pairs discussing the publisher in detail.

You will also be provided questions that already exist for the publisher, if any.
You should not repeat these, but build on them using the provided question suggestions.
It is vital you incorporate all information from the provided profile and make the
questions increasingly complex and long.

You should structure your response like so:

<begin_new_questions>
Question: question_1?
Answer: answer_1
...
Question: question_n?
Answer: answer_n
<end_new_questions>
"""


hallucinate_answer_pre_prompt = """
You have been tasked with generating an incorrect response to a question. You will not
know the answer or the context of the question, but you must generate a realistic
sounding answer. This answer will be used as an incorrect option in a multiple choice
setting.

You will receive a question, and you must generate your answer to the question and
nothing else. In your response it is imperative you do not reference the fact that it is
an incorrect answer, an hallucination, or anything otherwise. Your response should
contain only the text of the answer.
"""
