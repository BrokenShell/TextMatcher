from model import TextMatcher
from train_data import archetypes


def matcher():
    match = TextMatcher(archetypes, ngram_range=(1, 3), max_features=25000)

    def worker():
        description = input("\nDescription: ")
        while match(description) == 'No Match':
            description += '; ' + input("More data please: ")
        print(f"Match: {match(description)}\n")
        if not input("Would you like another? ").lower() in ('y', 'ya', 'yes'):
            exit()
        else:
            worker()

    worker()


if __name__ == '__main__':
    print("\nInteractive Archetypal Classification Example")
    print("Provide a description for me to match...")
    matcher()
