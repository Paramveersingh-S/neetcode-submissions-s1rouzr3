import torch
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive, negative):
        """
        Convert positive and negative sentences into a padded integer tensor.
        Words are assigned IDs sorted lexicographically (starting from 1).
        Positive rows come first, then negative rows, all padded to max length.

        Args:
            positive (list[str]): List of positive sentences.
            negative (list[str]): List of negative sentences.

        Returns:
            torch.Tensor: Shape (2*N, T) where N = len(positive) = len(negative)
                          and T is the maximum number of words among all sentences.
        """
        # Combine all sentences
        all_sentences = positive + negative

        # Build vocabulary: unique words sorted alphabetically
        unique_words = set()
        for sent in all_sentences:
            unique_words.update(sent.split())
        sorted_words = sorted(unique_words)
        word_to_id = {word: idx+1 for idx, word in enumerate(sorted_words)}  # start IDs at 1

        # Convert each sentence to a list of IDs
        sequences = []
        for sent in all_sentences:
            ids = [word_to_id[w] for w in sent.split()]  # empty list if sentence is empty
            sequences.append(torch.tensor(ids, dtype=torch.long))

        # Pad to the same length (0 is used for padding)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded