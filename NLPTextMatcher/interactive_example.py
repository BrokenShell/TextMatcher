""" This is an example of how the TextMatch can be used. The training data is
woefully inadequate for real-world work, and only provided here as inspiration.
Good training data would be five times larger, great training data would be
ten times larger or more. """
from NLPTextMatcher.model import TextMatcher
from NLPTextMatcher.train_data import archetypes

__all__ = ("matcher",)


def matcher():
    """ Interactive text matching script """
    match = TextMatcher(archetypes)

    def worker():
        description = input("\nDescription: ")
        while match(description) == "No Match":
            description += '; ' + input("More data please: ")
        print(f"Match: {match(description)}\n")
        if not input("Would you like another? ").lower() in ("y", "ya", "yes"):
            exit()
        else:
            worker()

    worker()


if __name__ == "__main__":
    print("\nInteractive Archetypal Classification Example")
    print("Provide a description of someone for me to match...")
    matcher()
