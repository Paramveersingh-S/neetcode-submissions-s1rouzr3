class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> list[list[str]]:
        # Step 1: Split corpus into individual characters
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            # If we have 1 or 0 tokens left, we can't merge anything
            if len(tokens) < 2:
                break

            # Step 2: Count frequencies of adjacent pairs
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            # Step 3: Find the most frequent pair (with lexicographical tie-breaker)
            best_pair = None
            max_count = -1

            for pair, count in pair_counts.items():
                if count > max_count:
                    max_count = count
                    best_pair = pair
                elif count == max_count:
                    # Tie-breaker: lexicographically smallest
                    if pair < best_pair:
                        best_pair = pair

            # If no pairs exist (should be caught by len check, but safe to have)
            if best_pair is None:
                break

            # Record the merge operation
            merges.append([best_pair[0], best_pair[1]])

            # Step 4: Replace non-overlapping occurrences (Left-to-Right)
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if we found the target pair
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2 # Skip the next token since it was merged
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            # Update our tokens list for the next iteration
            tokens = new_tokens

        return merges