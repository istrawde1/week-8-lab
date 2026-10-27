from collections import defaultdict
import numpy as np


class MarkovText(object):
    """Generate text from a corpus using a Markov chain."""

    def __init__(self, corpus):
        """Store the corpus and prepare the transition dictionary."""
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self):
        """Build a dictionary of words and their following words."""
        terms = self.corpus.split()
        term_dict = defaultdict(list)

        for current_word, next_word in zip(terms, terms[1:]):
            term_dict[current_word].append(next_word)

        self.term_dict = dict(term_dict)
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        """Generate up to term_count words from the corpus.

        Args:
            seed_term: Starting word. A random word is chosen if omitted.
            term_count: Maximum number of words to generate.

        Returns:
            The generated text as a string.

        Raises:
            ValueError: If the seed is not in the corpus.
        """
        terms = self.corpus.split()

        if seed_term is None:
            seed_term = np.random.choice(terms)
        elif seed_term not in terms:
            raise ValueError("Seed term is not in the corpus")

        generated = [seed_term]
        current_word = seed_term

        for _ in range(term_count - 1):
            next_words = self.term_dict.get(current_word)

            if not next_words:
                break

            current_word = np.random.choice(next_words)
            generated.append(current_word)
        
        return " ".join(generated)