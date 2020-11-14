""" This is an example of how the TextMatch can be used. The training data is
woefully inadequate for real-world work, and only provided here as inspiration.
Good training data would be five times larger, great training data would be
ten times larger or more. """
from TextMatcher.model import TextMatcher
from TextMatcher.train_data import archetypes
from TextMatcher.test_data import biographies

__all__ = ("celeb_classification",)


def celeb_classification():
    print("\nExample: Carl Jung Archetype Classification of Celebrities\n")
    matcher = TextMatcher(archetypes)
    for name, description in biographies.items():
        print(f"{name}: {matcher(description)}")


if __name__ == '__main__':
    celeb_classification()
