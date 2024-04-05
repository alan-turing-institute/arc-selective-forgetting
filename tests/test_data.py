import sys
from arcsf.utils.data_utils import load_tofu

def _test_load_tofu()
    granularity = sys.argv[1]
    print('Granularity:',granularity)

    forget_set, retain_set = load_tofu(granularity = granularity, forgotten_author_fraction = 0.1, random_seed = 42)

    print(forget_set)
    print(retain_set)
    n_to_test = 20

    for n,QA in enumerate(zip(forget_set['question'],forget_set['answer'])):
        if n >= n_to_test:
            break
        else:
            print('\nQuestion:',QA[0])
            print('Answer:',QA[1])

def test_load_tofu():
    options = ['random','random_within_authors','structured_within_authors','author_level']
    for granularity_index, granularity in enumerate(options):
        print(options)
    