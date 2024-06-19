import os

from openai import AzureOpenAI

# These are not used, but when I was transferring code over for question generation I
# tried to make these classes for question generation. I first focussed on building a
# a network with well defined connections. Once we know what our data needs to look
# like then we can start generating questions.


class BasicQuestionGenerator:
    """
    Classes for generating questions given a profile. This worked on the TOFU authors,
    but needs some adapting for the more complex profile setup.
    """

    def __init__(self, profile):
        """
        Args:
            profile: dict of the profile the question generator is being used for
        """
        self.profile = profile
        self.subject_map_q = {
            "d.o.b": f"What is {self.profile['Name']}'s date of birth?",
            "nationality": f"What is {self.profile['Name']}'s nationality?",
            "genre": f"What genre does {self.profile['Name']}'s write?",
            "publisher": f"What publisher does {self.profile['Name']} write for?",
            "books": f"What are some of {self.profile['Name']}'s published work?",
        }
        self.subject_map_a = {
            "d.o.b": (
                f"{self.profile['Name']}'s date of birth is {self.profile['D.O.B']}."
            ),
            "nationality": (
                f"{self.profile['Name']}'s nationality is "
                f"{self.profile['Nationality']}."
            ),
            "genre": (
                f"{self.profile['Name']} writes under the genre"
                f" of {self.profile['Genre']}."
            ),
            "publisher": (
                f"{self.profile['Name']} writes for {self.profile['Publisher']}."
            ),
            "books": (
                f"Some of {self.profile['Name']}'s include "
                f"{self.gen_book_list(self.profile['Books'])}."
            ),
        }

    def gen_book_list(self, books: list[str]) -> str:
        """
        This parses a list of books and returns the list formatted as a string.

        Args:
            books: list of books

        Returns:
            all books formatted as a string
        """
        book_string = books[0]
        for book in books[1:-1]:
            book_string += f", {book}"
        return book_string + f", and {books[-1]}"

    def __call__(self, q_subject: str) -> tuple[str]:
        """
        uses the mapping dicts to create a simple question answer pair when given an
        author and a question subject.

        Args:
            q_subject: string denoting the subject of a question

        Returns:
            tuple of a question and its answer
        """
        return self.subject_map_q[q_subject], self.subject_map_a[q_subject]


class QuestionGenerator:
    """
    This was some code I had written to get the GPT written questions going, its not
    really used for anything at the moment.
    """

    def __init__(self, profiles, default_pre_prompt=None):

        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("GPT_SANDBOX_ENDPOINT"),
            api_key=os.getenv("GPT_SANDBOX_KEY"),
            api_version="2024-02-01",
        )

        self.all_profiles = profiles
        self.basic_qs = ["d.o.b", "nationality", "genre", "publisher", "books"]

        if default_pre_prompt:
            self.base_chat = [{"role": "system", "content": default_pre_prompt}]

    def gen_book_list(self, books):
        book_string = books[0]
        for book in books[1:-1]:
            book_string += f", {book}"
        return book_string + f", and {books[-1]}"

    def gen_basic_questions(self, profile_index):
        generator = BasicQuestionGenerator(self.all_profiles[profile_index])
        qas = []
        for q_subject in ["d.o.b", "nationality", "genre", "publisher", "books"]:
            qas.append(generator(q_subject))
        return qas

    def gen_complex_questions(self, prompt):
        question = self.base_chat
        question.append([{"role": "user", "content": prompt}])

        response = self.client.chat.completions.create(
            model="gpt-forgetting", messages=question
        )
        return response.choices[0].message.content
