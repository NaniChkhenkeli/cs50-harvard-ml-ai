import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Get the number of pages in the corpus
    num_pages = len(corpus)
    
    # Initialize the probability distribution
    distribution = {p: 0 for p in corpus}
    
    # Get the links from the current page
    links = corpus[page]
    
    if links:
        # Probability of choosing a link from the current page
        prob_link = damping_factor / len(links)
        for link in links:
            distribution[link] += prob_link
    else:
        # If there are no links, treat it as if it links to all pages
        prob_link = 1 / num_pages
        for p in corpus:
            distribution[p] += prob_link
    
    # Probability of choosing any page at random
    prob_random = (1 - damping_factor) / num_pages
    for p in corpus:
        distribution[p] += prob_random
    
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize the count of visits to each page
    page_visits = {page: 0 for page in corpus}
    
    # Start with a random page
    current_page = random.choice(list(corpus.keys()))
    
    for _ in range(n):
        # Increment the visit count for the current page
        page_visits[current_page] += 1
        
        # Get the transition model for the current page
        transition_probs = transition_model(corpus, current_page, damping_factor)
        
        # Choose the next page based on the transition probabilities
        current_page = random.choices(list(transition_probs.keys()), weights=transition_probs.values())[0]
    
    # Normalize the visit counts to get PageRank values
    total_visits = sum(page_visits.values())
    return {page: visits / total_visits for page, visits in page_visits.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Number of pages in the corpus
    num_pages = len(corpus)
    
    # Initialize PageRank values
    ranks = {page: 1 / num_pages for page in corpus}
    
    # Continue iterating until convergence
    while True:
        new_ranks = {}
        for page in corpus:
            # Calculate the new PageRank for the page
            new_rank = (1 - damping_factor) / num_pages
            
            # Add contributions from all pages that link to this page
            for other_page in corpus:
                if page in corpus[other_page]:  # If other_page links to page
                    new_rank += damping_factor * (ranks[other_page] / len(corpus[other_page]))
                elif not corpus[other_page]:  # If other_page has no links
                    new_rank += damping_factor * (ranks[other_page] / num_pages)
            
            new_ranks[page] = new_rank
        
        # Check for convergence
        if all(abs(new_ranks[page] - ranks[page]) < 0.001 for page in ranks):
            break
        
        ranks = new_ranks
    
    return ranks


if __name__ == "__main__":
    main()