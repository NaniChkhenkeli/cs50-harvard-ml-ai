import sys
import nltk

# Ensure punkt tokenizer is downloaded
nltk.download('punkt')

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself" | "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she" | "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat" | "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S

NP -> N | Det N | Det AdjP N | Det N PP | Det AdjP N PP | NP PP

AdjP -> Adj | Adj AdjP

VP -> V | V NP | V NP PP | V PP | Adv VP | VP Adv

PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read().strip()  # Use .strip() to remove unwanted newlines or spaces
    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ").strip()

    # If sentence is empty, print an error
    if not s:
        print("No sentence provided.")
        return

    # Preprocess sentence
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(f"Parsing error: {e}")
        return

    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks:")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))

def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    tokens = nltk.word_tokenize(sentence.lower())  # Tokenize sentence and lowercase
    return [word for word in tokens if any(c.isalpha() for c in word)]  # Filter non-alphabetic words

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []

    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        # Add only those NP subtrees that do not have another NP as a child
        if not any(child.label() == "NP" for child in subtree.subtrees(lambda t: t != subtree)):
            chunks.append(subtree)

    return chunks

if __name__ == "__main__":
    main()
