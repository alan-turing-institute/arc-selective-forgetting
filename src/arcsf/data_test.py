from utils.data_utils import load_tofu

granularity = 'author_level'
granularity = 'fact_level_structured'

forget_set, retain_set = load_tofu(granularity = 'fact_level_structured', forgotten_author_fraction = 0.1, random_seed = 42)

print(forget_set)
print(retain_set)
n_to_test = 20

for n,QA in enumerate(zip(forget_set['question'],forget_set['answer'])):
    if n > n_to_test:
        break
    else:
        print('\nQuestion:',QA[0])
        print('Answer:',QA[1])