import random
import time
from uuid import uuid4


def find_between(string, start, end):
    return string.split(start)[1].split(end)[0]


def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, "%d/%m/%Y", prop)


def format_book(book, all_items):
    return (
        f"Book:\n"
        f"Name: {book['name']}\n"
        f"Author: {all_items[book['author']]['data']['name']}\n"
        f"Published: {book['published']}\n"
        f"Publisher: {all_items[book['publisher']]['data']['name']}"
    )


def format_author(author, all_items, country_map):
    return (
        f"Author:\n"
        f"Name: {author['name']}\n"
        f"Date of Birth: {author['dob']}\n"
        f"Nationality: {country_map[all_items[author['nationality']]['data']['name']]}"
    )


def format_publisher(publisher, all_items):
    return (
        f"Publisher:\n"
        f"Name: {publisher['name']}\n"
        f"Founded: {publisher['founded']}\n"
        f"Based in: {all_items[publisher['country']]['data']['name']}"
    )


def get_book_connections(book):
    return [
        (book["key"], book["author"]),
        (book["key"], book["publisher"]),
        (book["key"], book["genre"]),
    ]


def get_author_connections(author):
    return [(author["key"], author["nationality"])]


def get_publisher_connections(publisher):
    return [(publisher["key"], publisher["country"])]


class Formatter:

    def __init__(self, all_items, country_map):
        self.all_items = all_items
        self.country_map = country_map

    def print_item(self, key):
        item = self.all_items[key]
        if item["type"] == "book":
            return format_book(item["data"], self.all_items)
        elif item["type"] == "author":
            return format_author(item["data"], self.all_items, self.country_map)
        elif item["type"] == "publisher":
            return format_publisher(item["data"], self.all_items)
        else:
            return f"{item['type'].capitalize()}: {item['data']['name'].capitalize()}"

    def get_connections(self, key):
        item = self.all_items[key]
        if item["type"] == "book":
            return get_book_connections(item["data"])
        elif item["type"] == "author":
            return get_author_connections(item["data"])
        elif item["type"] == "publisher":
            return get_publisher_connections(item["data"])
        else:
            return []


class Sampler:

    def __init__(self, distribution_dict):
        self.options = distribution_dict["options"]
        self.indices = range(0, len(self.options))
        self.distribution = distribution_dict["distribution"]

    def sample(self):
        if sum(self.distribution) <= 0:
            raise ValueError("Max number of samples exceeded")
        choice = random.choices(self.indices, self.distribution, k=1)[0]
        self.distribution[choice] -= 1
        return self.options[choice]


class PublisherSampler:

    def __init__(self, country_dist, date_limits):
        self.country_sampler = Sampler(country_dist)
        self.date_limits = date_limits

    def sample(self, name):
        profile = {}
        profile["key"] = str(uuid4())
        profile["name"] = name
        profile["country"] = self.country_sampler.sample()
        profile["founded"] = random_date(*self.date_limits, random.random())
        return profile


class AuthorSampler:

    def __init__(self, country_dist, genre_dist, date_limits):
        self.country_sampler = Sampler(country_dist)
        self.genre_sampler = Sampler(genre_dist)
        self.date_limits = date_limits

    def sample(self):
        profile = {}
        profile["key"] = str(uuid4())
        profile["name"] = "placeholder"
        profile["dob"] = random_date(*self.date_limits, random.random())
        profile["nationality"] = self.country_sampler.sample()
        profile["genre"] = self.genre_sampler.sample()
        return profile


class BookSampler:

    def __init__(self, publisher_dist, author_dist, date_limits):
        self.publisher_sampler = Sampler(publisher_dist)
        self.author_sampler = Sampler(author_dist)
        self.date_limits = date_limits

    def sample(self):
        author = self.author_sampler.sample()
        publisher = self.publisher_sampler.sample()
        profile = {}
        profile["key"] = str(uuid4())
        profile["name"] = "placeholder"
        profile["author"] = author["key"]
        profile["publisher"] = publisher["key"]
        profile["genre"] = author["genre"]

        date_limits = self.date_limits
        lower_limit = time.strftime(
            "%d/%m/%Y",
            max(
                time.strptime(publisher["founded"], "%d/%m/%Y"),
                time.strptime(author["dob"], "%d/%m/%Y"),
            ),
        )
        date_limits[0] = lower_limit
        profile["published"] = random_date(*date_limits, random.random())

        return profile
