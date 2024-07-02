import os

from openai import AzureOpenAI

# These are not used, but when I was transferring code over for question generation I
# tried to make these classes for question generation. I first focussed on building a
# a network with well defined connections. Once we know what our data needs to look
# like then we can start generating questions.


def entity_qa_generator(entity):
    entity_type = entity["type"]
    entity_data = entity["data"]
    if entity_type == "publisher":
        question = f"When was {entity_data['name'].capitalize()} founded?"
        answer = (
            f"{entity_data['name'].capitalize()} was founded on"
            f" {entity_data['founded']}."
        )
        return (question, answer)
    elif entity_type == "author":
        question = f"When was {entity_data['name'].capitalize()} born?"
        answer = (
            f"{entity_data['name'].capitalize()}'s date of birth is"
            f" {entity_data['dob']}."
        )
        return (question, answer)
    elif entity_type == "book":
        question = f"When was {entity_data['name'].capitalize()} published?"
        answer = (
            f"{entity_data['name'].capitalize()} was published on"
            f" {entity_data['published']}."
        )
        return (question, answer)
    else:
        return None


def relationship_qa_generator(main_entity, relationship_entity):
    main_type = main_entity["type"]
    main_entity_data = main_entity["data"]
    relation_type = relationship_entity["type"]
    relation_entity_data = relationship_entity["data"]
    if main_type == "publisher":
        if relation_type == "country":
            question = (
                f"In what country is the publisher"
                f" {main_entity_data['name'].capitalize()} based?"
            )
            answer = (
                f"{main_entity_data['name'].capitalize()} is based in "
                f" {relation_entity_data['name']}."
            )
        return (question, answer)

    elif main_type == "author":
        if relation_type == "country":
            question = (
                f"Where was the author,"
                f" {main_entity_data['name'].capitalize()}, born?"
            )
            answer = (
                f"{main_entity_data['name'].capitalize()} was born in "
                f" {relation_entity_data['name']}."
            )
        return (question, answer)

    elif main_type == "book":
        if relation_type == "author":
            question = (
                f"Who wrote the book," f" {main_entity_data['name'].capitalize()}?"
            )
            answer = (
                f"{main_entity_data['name'].capitalize()} was written by"
                f" {relation_entity_data['name']}."
            )
        elif relation_type == "publisher":
            question = (
                f"Which publisher published the book,"
                f" {main_entity_data['name'].capitalize()}?"
            )
            answer = (
                f"{main_entity_data['name'].capitalize()} was published by"
                f" {relation_entity_data['name']}."
            )
        elif relation_type == "genre":
            question = (
                f"What genre is the book," f" {main_entity_data['name'].capitalize()}?"
            )
            answer = (
                f"{main_entity_data['name'].capitalize()} falls under the genre of"
                f" {relation_entity_data['name']} books."
            )
        return (question, answer)
    else:
        return None


def list_names(entities):
    n_entities = len(entities)
    string = entities[0]["name"]
    if n_entities == 1:
        return string
    elif n_entities == 2:
        return string + f" and {entities[-1]['name']}"
    else:
        for entity in entities[1:-1]:
            string += f", {entity['name']}"
        return string + f", and {entities[-1]['name']}"


def relationship_list_qa_generator(main_entity, related_entities):
    main_type = main_entity["type"]
    main_entity_data = main_entity["data"]

    related_entity_type = related_entities[0]["type"]
    related_entity_data = [entity["data"] for entity in related_entities]

    if main_type == "publisher":
        if related_entity_type == "book":
            question = (
                f"What are some books published by "
                f"{main_entity_data['name'].capitalize()}?"
            )
            answer = (
                f"Some books published by {main_entity_data['name'].capitalize()} "
                f"include: {list_names(related_entity_data)}."
            )
        return (question, answer)

    elif main_type == "author":
        if related_entity_type == "book":
            question = (
                f"What are some books written by "
                f"{main_entity_data['name'].capitalize()}?"
            )
            answer = (
                f"Some books written by {main_entity_data['name'].capitalize()} "
                f"include: {list_names(related_entity_data)}."
            )
        return (question, answer)


class NetworkQuestionGenerator:
    def __init__(self, all_profiles, all_connections):
        self.all_profiles = all_profiles
        self.all_connections = all_connections

    def sample_basic_question(self, key: str) -> tuple[str]:
        profile = self.all_profiles[key]
        # generating simple question
        qa_pair = entity_qa_generator(profile)
        return qa_pair

    def sample_relationship_question(self, keys: tuple[str, str]) -> tuple[str]:
        main_profile = self.all_profiles[keys[0]]
        relationship_profile = self.all_profiles[keys[1]]
        qa_pair = relationship_qa_generator(main_profile, relationship_profile)
        return qa_pair

    def sample_relationship_list_question(
        self, key: str, target_type: str
    ) -> tuple[str]:
        main_profile = self.all_profiles[key]
        related_profiles = []
        qa_keys = [key]
        for connection in self.all_connections:
            if key in connection:
                for related_key in connection:
                    if self.all_profiles[related_key]["type"] == target_type:
                        related_profiles.append(self.all_profiles[related_key])
                        qa_keys.append(related_key)

        qa_pair = relationship_list_qa_generator(main_profile, related_profiles)
        return qa_pair, qa_keys


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
