import random
import time
from typing import Any
from uuid import uuid4


def flatten(xss):
    """
    flattens a list

    Args:
        xss: list of nested lists

    Returns:
        flattened list
    """
    return [x for xs in xss for x in xs]


def find_between(string: str, start: str, end: str):
    """
    Extracts the string between a start and end tag within a string.

    Args:
        string: input string
        start: start tag
        end: end tag

    Returns:
        extracted string
    """
    return string.split(start)[1].split(end)[0]


def str_time_prop(start, end, time_format, diff):
    """
    Get a time at a difference in a range between two formatted times.
    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.

    Args:
        start: start date
        end: end date
        time_format: format for date string
        diff: random difference to define the date object

    Returns:
        random date
    """
    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + diff * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start: str, end: str, diff: float) -> str:
    """
    Selects a random date within a defined range.

    Args:
        start: start date
        end: end date
        diff: random difference

    Returns:
        date in string format
    """
    return str_time_prop(start, end, "%d/%m/%Y", diff)


def format_book(book: dict[str:str], all_items: dict[dict[str:str]]) -> str:
    """
    Formats a book item into a string for use in data generation.

    Args:
        book: book item within the dataset
        all_items: dictionary containing all items

    Returns:
        Formatted string of relevant entries for the item.
    """
    return (
        f"Book:\n"
        f"Name: {book['name']}\n"
        f"Genre: {all_items[book['genre']]['data']['name']}\n"
        f"Author: {all_items[book['author']]['data']['name']}\n"
        f"Published: {book['published']}\n"
        f"Publisher: {all_items[book['publisher']]['data']['name']}"
    )


def format_author(
    author: dict[str:str], all_items: dict[dict[str:str]], country_map: dict[str:str]
):
    """
    Formats am author item into a string for use in data generation.

    Args:
        author: author item within the dataset
        all_items: dictionary containing all items
        country_map: dictionary mapping country proper noun to adjective
        (eg. England -> English)

    Returns:
        Formatted string of relevant entries for the item.
    """
    return (
        f"Author:\n"
        f"Name: {author['name']}\n"
        f"Date of Birth: {author['dob']}\n"
        f"Nationality: {country_map[all_items[author['nationality']]['data']['name']]}"
        f"\nGenre: {all_items[author['genre']]['data']['name']}"
        f"\nNumber of siblings: {all_items[author['siblings']]['data']['name']}"
        f"\nParent relationship:"
        f" {all_items[author['parent_relationship']]['data']['name']}"
        f"\nFormer career: {all_items[author['career']]['data']['name']}"
        f"\nFormal education: {all_items[author['education']]['data']['name']}"
    )


def format_publisher(publisher, all_items):
    """
    Formats a publisher item into a string for use in data generation.

    Args:
        publisher: publisher item within the dataset
        all_items: dictionary containing all items

    Returns:
        Formatted string of relevant entries for the item.
    """
    return (
        f"Publisher:\n"
        f"Name: {publisher['name']}\n"
        f"Founded: {publisher['founded']}\n"
        f"Based in: {all_items[publisher['country']]['data']['name']}"
    )


def get_book_connections(book: dict[str:str]) -> list[tuple[str]]:
    """
    Return a list of connections associated with a book item.

    Args:
        book: book item in the network

    Returns:
        list of tuples denoting network links
    """
    return [
        (book["key"], book["author"]),
        (book["key"], book["publisher"]),
        (book["key"], book["genre"]),
    ]


def get_author_connections(author: dict[str:str]) -> list[tuple[str]]:
    """
    Return a list of connections associated with an author item.

    Args:
        author: author item in the network

    Returns:
        list of tuples denoting network links
    """
    return [(author["key"], author["nationality"])]


def get_publisher_connections(publisher: dict[str:str]) -> list[tuple[str]]:
    """
    Return a list of connections associated with a publisher item.

    Args:
        publisher: publisher item in the network

    Returns:
        list of tuples denoting network links
    """
    return [(publisher["key"], publisher["country"])]


def get_other_connections(entity_key: str, all_items: dict[str:dict]):
    connections = []
    for key, item in all_items.items():
        if entity_key in item["data"].values():
            if key != entity_key:
                connections.append((entity_key, key))
    return connections


class Formatter:
    """
    Class for formatting and retrieving connections of items in the network.
    """

    def __init__(
        self, all_items: dict[str : dict[str:str]], country_map: dict[str:str]
    ):
        """
        Args:
            all_items: full network dictionary containing all items
            country_map: mapping rules for countries to associated adjectives
            (eg. England -> English)
        """
        self.all_items = all_items
        self.country_map = country_map

    def print_item(self, key: str) -> str:
        """
        Formats an item in the dictionary into readable string format for data
        generation

        Args:
            key: UUID key denoting the item in the dictionary

        Returns:
            String formatting the item into a readable format.
        """
        item = self.all_items[key]
        if item["type"] == "book":
            return format_book(item["data"], self.all_items)
        elif item["type"] == "author":
            return format_author(item["data"], self.all_items, self.country_map)
        elif item["type"] == "publisher":
            return format_publisher(item["data"], self.all_items)
        else:
            return f"{item['type'].capitalize()}: {item['data']['name'].capitalize()}"

    def get_connections(self, key: str, other_flag: bool = False) -> list[tuple]:
        """
        Returns the connections associated with an item.

        Args:
            key: UUID key denoting the item in the dictionary

        Returns:
            List of tuples denoting associated connections with the item. If item is at
            the top of the hierarchy (eg. Country), empty list is returned.
        """
        item = self.all_items[key]
        if item["type"] == "book":
            return get_book_connections(item["data"])
        elif item["type"] == "author":
            return get_author_connections(item["data"])
        elif item["type"] == "publisher":
            return get_publisher_connections(item["data"])
        else:
            if other_flag:
                return get_other_connections(key, self.all_items)
            return []


class Sampler:
    """
    Class for sampling elements randomly with a predefined number counts on each.
    """

    def __init__(self, distribution_dict: dict[str : list[Any] | int]):
        """
        Args:
            distribution_dict: dictionary denoting the elements in the 'options' key and
            and the predefined counts per item in the 'distribution' key.
        """
        self.options = distribution_dict["options"]
        self.indices = range(0, len(self.options))
        self.distribution = distribution_dict["distribution"]

    def sample(self):
        """
        Randomly samples an item from the options list

        Raises:
            ValueError: If all elements have been sampled their predifined number of
            times

        Returns:
            Randomly selected item from the list of elements
        """
        if sum(self.distribution) <= 0:
            raise ValueError("Max number of samples exceeded")
        choice = random.choices(self.indices, self.distribution, k=1)[0]
        # once sampled reduce the number of times it should be sampled
        self.distribution[choice] -= 1
        return self.options[choice]


class PublisherSampler:
    """
    Class for randomly sampling publishers
    """

    def __init__(
        self, country_dist: dict[str : list[str], list[int]], date_limits: list[str]
    ):
        """
        Args:
            country_dist: Country options for the publisher to be based in
            date_limits: Date limits for the publisher to be founded
        """
        self.country_sampler = Sampler(country_dist)
        self.date_limits = date_limits

    def sample(self, name: str) -> dict[str:str]:
        """
        Randomly sample the properties for a predefined author name.

        Args:
            name: Publisher name

        Returns:
            profile: dictionary containing the publisher profile
        """
        profile = {}
        profile["key"] = str(uuid4())
        profile["name"] = name
        profile["country"] = self.country_sampler.sample()
        profile["founded"] = random_date(*self.date_limits, random.random())
        return profile


class AuthorSampler:
    """
    Class for randomly sampling authors
    """

    def __init__(
        self,
        country_dist: dict[str : list[str], list[int]],
        genre_dist: dict[str : list[str], list[int]],
        date_limits: list[str],
    ):
        """
        Args:
            country_dist: Nationality options for the author
            genre_dist: Genre options for the author's books
            date_limits: Date limits for the author date of birth
        """
        self.country_sampler = Sampler(country_dist)
        self.genre_sampler = Sampler(genre_dist)
        self.date_limits = date_limits
        # this is used to create unique names for each entity
        self.sample_counts = 0

    def sample(self) -> dict[str:str]:
        """
        Randomly sample an author object for the network.

        Returns:
            profile: dictionary containing the author profile
        """
        profile = {}
        profile["key"] = str(uuid4())
        profile["name"] = f"author_{self.sample_counts}"
        profile["dob"] = random_date(*self.date_limits, random.random())
        profile["nationality"] = self.country_sampler.sample()
        profile["genre"] = self.genre_sampler.sample()
        self.sample_counts += 1
        return profile


class BookSampler:
    """
    Class for randomly sampling books
    """

    def __init__(
        self,
        publisher_dist: dict[str : list[str], list[int]],
        author_dist: dict[str : list[str], list[int]],
        date_limits: list[str],
    ):
        """
        Args:
            publisher_dist: Publisher options for the book
            author_dist: Author options for the book
            date_limits: General date limits defined for books
        """
        self.publisher_sampler = Sampler(publisher_dist)
        self.author_sampler = Sampler(author_dist)
        self.date_limits = date_limits
        # this is used to create unique names for each entity
        self.sample_counts = 0

    def sample(self) -> dict[str:str]:
        """
        Randomly sample a book object for the network.

        Returns:
            profile: dictionary containing a book profile
        """
        author = self.author_sampler.sample()
        publisher = self.publisher_sampler.sample()
        profile = {}
        profile["key"] = str(uuid4())
        profile["name"] = f"book_{self.sample_counts}"
        profile["author"] = author["key"]
        profile["publisher"] = publisher["key"]
        profile["genre"] = author["genre"]

        # ensuring books are published after publisher founded and after authors are
        # old enough (ish) to write books, manipulating string was easier than messing
        # around with the time() objects...
        date_limits = self.date_limits
        author_year = str(int(author["dob"][-4:]) + 18)
        author_publish = author["dob"][:-4] + author_year
        lower_limit = time.strftime(
            "%d/%m/%Y",
            max(
                time.strptime(publisher["founded"], "%d/%m/%Y"),
                time.strptime(author_publish, "%d/%m/%Y"),
            ),
        )
        date_limits[0] = lower_limit
        profile["published"] = random_date(*date_limits, random.random())
        self.sample_counts += 1

        return profile


class KeyChecker:
    """
    On call checks a dataset rows' keys against another list of target keys, returns
    true if a pair of keys match, otherwise returns False (This bool is flipped if
    find_forget is set as False).
    """

    def __init__(self, forget_keys: list[int], find_forget: bool):
        """
        Args:
            forget_keys: Keys that have been assigned to the forget set
            find_forget: Whether the output should be denoting forget or retain set.

        Raises:
            ValueError: If find_forget is assigned a non-bool value, a ValueError is
            outputted.
        """
        self.forget_keys = forget_keys
        if find_forget:
            self.output = lambda x: x
        elif not find_forget:
            self.output = lambda x: not x
        else:
            raise ValueError("find_forget must be set to True or False")

    def __call__(self, item: dict[str : str | list[int]]) -> bool:
        keys = item["keys"]
        flag = False
        for key in keys:
            if key in self.forget_keys:
                flag = True
        return self.output(flag)
