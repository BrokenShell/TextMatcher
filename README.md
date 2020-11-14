# TextMatcher
Author: Robert Sharp

### TextMatcher/model.py

#### TextMatcher: Primary Interface
Experimental machine learning model featuring natural language processing 
trained on arbitrary text for the purpose of topic classification of arbitrary 
text. The class creates a callable object for making predictions. The instance 
is trained at initialization and the callable object is reusable for the same 
training data. TextMatcher uses a combination of SpaCy, TfidfVectorizer 
and NearestNeighbors.

`__init__(train_data: dict, ngram_range: tuple, max_features: int) -> Callable`
- @param train_data: Dictionary of targets and supporting data. See
training_data.py for an example.
- @param ngram_range: Tuple representing the range of phrase sizes that
the model will recognize.
- @param max_features: The maximum number of tokens, this is used to
fine tune the maximum amount of RAM that the model is allowed to use
for training.

`__call__(user_input: str) -> str`
- @param user_input: Arbitrary string of text to be classified.
- @return: Name of the predicted classification target as a string.

#### Tokenizer: Helper Class
Creates a callable object for tokenizing input data based on the en_core_web_sm 
SpaCy library.

`__call__(text: str) -> list`
- @param text: String of text to be tokenized 
- @return: List of SpaCy tokens

### TextMatcher/interactive_example.py
An example that classifies user input by their nearest Carl Jung Archetype.
Execute this script as main for an interactive example.

### TextMatcher/celeb_example.py
- `celeb_classification()`
An example that classifies celebrities by their nearest Carl Jung Archetype.

### TextMatcher/test_data.py
Dictionary of biographies to by used in the celebrity classification example.

### TextMatcher/train_data.py
Carl Jung Archetype training data for the examples.
- The Hero
- The Caregiver
- The Explorer
- The Rebel
- The Lover
- The Artist
- The Entertainer
- The Sage
- The Magician
- The Ruler
