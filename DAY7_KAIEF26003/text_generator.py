import math
import random
from collections import defaultdict, Counter


class NGramLanguageModel:
    def __init__(self, n=2, alpha=1.0):
        self.n = n
        self.alpha = alpha  # Laplace smoothing parameter
        self.model = defaultdict(Counter)
        self.vocab = set()

    def train(self, corpus):
        for sentence in corpus:
            tokens = ["<s>"] * (self.n - 1) + sentence.lower().split() + ["</s>"]
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i : i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.model[context][next_word] += 1

    def get_word_prob(self, context, word):
        context_tuple = tuple(context)
        context_counts = self.model[context_tuple]
        total_count = sum(context_counts.values()) + self.alpha * len(self.vocab)
        word_count = context_counts.get(word, 0) + self.alpha
        return word_count / total_count

    def compute_perplexity(self, test_corpus):
        log_prob_sum = 0.0
        total_tokens = 0

        for sentence in test_corpus:
            tokens = ["<s>"] * (self.n - 1) + sentence.lower().split() + ["</s>"]
            for i in range(self.n - 1, len(tokens)):
                context = tokens[i - (self.n - 1) : i]
                word = tokens[i]
                prob = self.get_word_prob(context, word)
                log_prob_sum += math.log(prob)
                total_tokens += 1

        if total_tokens == 0:
            return float("inf")

        return math.exp(-log_prob_sum / total_tokens)

    def generate_text(self, max_length=20):
        context = ["<s>"] * (self.n - 1)
        generated = []

        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n - 1):])
            if context_tuple not in self.model or not self.model[context_tuple]:
                break

            words = list(self.model[context_tuple].keys())
            weights = list(self.model[context_tuple].values())

            next_word = random.choices(words, weights=weights)[0]
            if next_word == "</s>":
                break

            generated.append(next_word)
            context.append(next_word)

        return " ".join(generated)


def main():
    corpus = [
        "artificial intelligence is transforming modern technology and society",
        "deep learning models require large amounts of quality data",
        "language models predict the next word in a sequence of tokens",
        "natural language processing enables machines to comprehend human speech",
        "artificial intelligence models leverage statistics and probability theory",
        "deep learning architectures like transformers power intelligent agents",
        "learning representations from data is fundamental to modern machine learning"
    ]

    test_corpus = [
        "artificial intelligence uses deep learning models",
        "natural language processing transforms technology"
    ]

    print("--- Training Bigram & Trigram Language Models ---")

    # 1. Bigram Model
    bigram_model = NGramLanguageModel(n=2, alpha=1.0)
    bigram_model.train(corpus)
    bigram_ppl = bigram_model.compute_perplexity(test_corpus)

    print("\n[Bigram Model (N=2)]")
    print(f"Test Perplexity: {bigram_ppl:.2f}")
    print("Generated Text:", bigram_model.generate_text(max_length=15))

    # 2. Trigram Model
    trigram_model = NGramLanguageModel(n=3, alpha=1.0)
    trigram_model.train(corpus)
    trigram_ppl = trigram_model.compute_perplexity(test_corpus)

    print("\n[Trigram Model (N=3)]")
    print(f"Test Perplexity: {trigram_ppl:.2f}")
    print("Generated Text:", trigram_model.generate_text(max_length=15))


if __name__ == "__main__":
    main()