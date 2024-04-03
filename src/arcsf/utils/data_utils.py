import itertools 
import numpy as np 
from datasets import Dataset,load_dataset

def get_indices_structured(forgotten_author_numbers:list, q_remove:int = 4, n_authors:int = 200, q_per_author:int=20):
    '''
    Returns two flattened lists of Question--Answer indices given authors numbers to remove/retain, in the case where the forget target can be easily structured. Assumes equal questions per author, and sorted authors.

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

def get_author_splits(n_authors:int = 200, author_forget_fraction:float = 0.1, random_seed:int = 42):
    '''
    Returns randomly selected author numbers to retain/forget.

        Parameters:
                n_authors (list):       Total number of authors
                forget_fraction (float):Fraction of authors that should be removed
                random_seed (int):      Random seed for reproducibility

        Returns:
                forget_authors (list):  List of integers containing author indexes to remove/forget
    '''
    rng = np.random.default_rng(random_seed)
    author_options = np.arange(0,n_authors)
    forget_authors = np.sort(rng.choice(author_options,int(author_forget_fraction*n_authors),replace=False))

    return forget_authors


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
    author_q_count = 20                                         # hard coding author question count for now

    forget_author_numbers = get_author_splits(author_count,forgotten_author_fraction,random_seed)
    if granularity == 'author_level':
        forget_indices, retain_indices = get_indices_structured(forget_author_numbers,q_remove=author_q_count,n_authors=author_count,q_per_author=author_q_count)

    if granularity == 'fact_level_structured': 
        #biographical information generally contained in first 4 questions
        forget_indices, retain_indices = get_indices_structured(forget_author_numbers,q_remove=4,n_authors=author_count,q_per_author=author_q_count)

    forget_set,retain_set = Dataset.from_dict(all_data[forget_indices]),Dataset.from_dict(all_data[retain_indices])

    return forget_set,retain_set
