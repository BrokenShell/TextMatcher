from model import TextMatcher
from train_data import archetypes
from test_data import biographies


print("\nArchetypal Classification")
match = TextMatcher(archetypes, ngram_range=(1, 2), max_features=2500)
for name, description in biographies.items():
    print(f"{name}: {match(description)}")
print()
