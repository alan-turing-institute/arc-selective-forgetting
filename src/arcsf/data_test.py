from utils.data_utils import load_tofu

forget_set, retain_set = load_tofu(granularity = 'author_level', forget_fraction = 0.1, random_seed = 42)

print(forget_set)
print(retain_set)