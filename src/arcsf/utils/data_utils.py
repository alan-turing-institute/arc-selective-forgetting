import itertools 
import numpy as np 
from datasets import Dataset,load_dataset


def get_indices(author_numbers:list, q_per_author:int=20):
    '''
    Returns a flattened list of Question--Answer indices given authors numbers to remove. Assumes equal questions per author, and sorted authors.

        Parameters:
                author_numbers (list):  List of integers containing author indexes
                q_per_author (int):     Integer defining the number of questions per author

        Returns:
                stacked_indices (list): List of integers corresponding to the question indices pertaining the authors in author_numbers
    '''
    indices_list = list()
    for author_n in author_numbers:
        ref_index = author_n*q_per_author #start index for author questions
        indices_list.append(np.arange(ref_index,ref_index+q_per_author))
    stacked_indices = list(itertools.chain.from_iterable(indices_list))
    return stacked_indices

def get_indices_q_structured(forgotten_author_numbers:list, q_remove:int = 4, n_authors:int = 200, q_per_author:int=20):
    '''
    Returns two flattened lists of Question--Answer indices given authors numbers to remove/retain. Assumes equal questions per author, and sorted authors.

        Parameters:
                forgotten_author_numbers (list):    List of integers containing author indexes about which facts should be removed
                n_authors (int):                    Total number of authors
                q_removal (int):                    Number of questions/facts that should be removed
                q_per_author (int):                 Integer defining the number of questions to remove per author

        Returns:
                forget_indices (list):              List of integers corresponding to the question indices pertaining the authors in author_numbers
                retain_indices (list):              List of integers corresponding to the remaining question indices
    '''
    assert q_per_author <= 20, "Cannot remove more questions for an author than there are questions per author."
    indices_list = list()
    for author_n in forgotten_author_numbers:
        ref_index = author_n*q_per_author #start index for author questions
        indices_list.append(np.arange(ref_index,ref_index+q_remove))
    forget_indices = list(itertools.chain.from_iterable(indices_list))
    retain_indices = np.delete(np.arange(0,n_authors*q_per_author),forget_indices)
    return forget_indices, retain_indices

def get_splits(n_authors:int = 200, forget_fraction:float = 0.1, random_seed:int = 42):
    '''
    Returns randomly selected author numbers to retain/forget.

        Parameters:
                n_authors (list):       Total number of authors
                forget_fraction (float):Fraction of authors that should be removed
                random_seed (int):      Random seed for reproducibility

        Returns:
                forget_authors (list):  List of integers containing author indexes to remove/forget
                retain_authors (list):  List of integers containing author indexes to retain
    '''
    rng = np.random.default_rng(random_seed)
    author_options = np.arange(0,n_authors)
    forget_authors = np.sort(rng.choice(author_options,int(forget_fraction*n_authors),replace=False))
    retain_authors = np.delete(author_options,forget_authors)

    return forget_authors,retain_authors


def load_tofu(granularity:str = 'author_level', forgotten_author_fraction:float = 0.1, forgotten_fact_fraction:float = 0.1, random_seed:int = 42):
    '''
    Loads the TOFU dataset with randomly chosen authors/information to forget, using question indices to remove/retain data.

        Parameters:
                granularity (str):                          Granularity at which to perform unlearning
                forgotten_author_fraction (float):          Fraction of authors that should be removed
                forgotten_fact_fraction (float):            Fraction of facts to be removed (currently not used)
                random_seed (int):                          Random seed for reproducibility

        Returns:
                forget_set (datasets.arrow_dataset.Dataset):      Dataset containing removed/forgotten authors 
                retain_set (datasets.arrow_dataset.Dataset):      Dataset containing retained authors
    '''

    all_data = load_dataset("locuslab/TOFU", 'full')['train']   # load all data to work with
    author_count = 200                                          # hard coding author count for now

    forget_author_numbers, retain_authors_numbers = get_splits(author_count,forgotten_author_fraction,random_seed)
    if granularity == 'author_level':
        forget_author_numbers, retain_authors_numbers = get_splits(author_count,forgotten_author_fraction,random_seed)
        forget_indices, retain_indices = get_indices(forget_author_numbers),get_indices(retain_authors_numbers)

    if granularity == 'fact_level_structured':
        #for this we need to define the questions to remove, then all other questions remain, hence retain_authors_numbers is redundant 
        forget_author_numbers,_ = get_splits(author_count,forgotten_author_fraction,random_seed)
        #biographical information generally contained in first 4 questions
        forget_indices, retain_indices = get_indices_q_structured(forget_author_numbers,q_remove=4,n_authors=author_count,q_per_author=20)

    forget_set,retain_set = Dataset.from_dict(all_data[forget_indices]),Dataset.from_dict(all_data[retain_indices])

    return forget_set,retain_set
