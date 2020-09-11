from model import TextMatcher
from train_data import archetypes


match = TextMatcher(archetypes)

name = input("\nName: ")
description = input("Description: ")
while match(description) == 'No Match':
    description += '; ' + input("More data please: ")
print(f"{name}: {match(description)}\n")
if not input("Would you like another? ").lower() in ('y', 'ya', 'yes'):
    exit()
