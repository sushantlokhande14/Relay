"""Prompt corpus for the load test. Each topic has a canonical question and a
couple of close paraphrases. The driver mixes exact repeats (which become exact
cache hits), paraphrases (which become semantic hits when they clear the
threshold), and unique throwaway prompts (which always miss)."""
from __future__ import annotations

import random

TOPICS = [
    {"q": "what is the capital of France?",
     "p": ["what's the capital city of France?", "tell me the capital of France"]},
    {"q": "how do I reverse a list in python?",
     "p": ["how can I reverse a list in python", "what's the way to reverse a python list?"]},
    {"q": "what is the speed of light?",
     "p": ["how fast does light travel?", "what's the speed of light in a vacuum?"]},
    {"q": "who wrote pride and prejudice?",
     "p": ["who is the author of pride and prejudice?", "pride and prejudice was written by whom?"]},
    {"q": "how does photosynthesis work?",
     "p": ["explain how photosynthesis works", "what is the process of photosynthesis?"]},
    {"q": "what is the boiling point of water?",
     "p": ["at what temperature does water boil?", "what's the boiling temperature of water?"]},
    {"q": "what is a binary search tree?",
     "p": ["explain what a binary search tree is", "what's a BST in data structures?"]},
    {"q": "how do I center a div in css?",
     "p": ["what's the way to center a div with css?", "how can I center a div using css"]},
    {"q": "what causes the seasons on earth?",
     "p": ["why does the earth have seasons?", "what makes the seasons change on earth?"]},
    {"q": "what is the difference between tcp and udp?",
     "p": ["how do tcp and udp differ?", "compare tcp and udp"]},
    {"q": "how do I make an http request in python?",
     "p": ["what's the way to send an http request in python?", "how can I do an http call in python"]},
    {"q": "what is machine learning?",
     "p": ["can you explain machine learning?", "what does machine learning mean?"]},
    {"q": "what is the tallest mountain in the world?",
     "p": ["which mountain is the tallest on earth?", "what's the highest mountain in the world?"]},
    {"q": "how does a hash table work?",
     "p": ["explain how a hash table works", "what's the mechanism behind a hash table?"]},
    {"q": "what is the chemical formula for water?",
     "p": ["what's water's chemical formula?", "give the chemical formula of water"]},
    {"q": "what is recursion in programming?",
     "p": ["explain recursion in programming", "what does recursion mean in code?"]},
    {"q": "who painted the mona lisa?",
     "p": ["who is the painter of the mona lisa?", "the mona lisa was painted by whom?"]},
    {"q": "how do I sort a dictionary by value in python?",
     "p": ["what's the way to sort a python dict by its values?", "how can I order a dictionary by value in python"]},
    {"q": "what is the largest planet in the solar system?",
     "p": ["which planet is the biggest in our solar system?", "what's the largest planet orbiting the sun?"]},
    {"q": "what is an api?",
     "p": ["what does api mean?", "can you explain what an api is?"]},
    {"q": "how does https keep data secure?",
     "p": ["what makes https secure?", "how does https protect data in transit?"]},
    {"q": "what is the freezing point of water?",
     "p": ["at what temperature does water freeze?", "what's the freezing temperature of water?"]},
    {"q": "what is a docker container?",
     "p": ["explain what a docker container is", "what does a docker container do?"]},
    {"q": "how do I read a file line by line in python?",
     "p": ["what's the way to read a file line by line in python?", "how can I iterate over file lines in python"]},
]


# Vocabulary for genuinely novel "miss" prompts. A random handful of unrelated
# words from a broad list embeds far from anything else in the cache, so these
# stay misses instead of accidentally matching each other (which a templated
# generator does).
VOCAB = (
    "river copper anxious telescope harvest umbrella glacier velvet senator molecule "
    "saxophone meadow tungsten nostalgia compiler avalanche cinnamon turbine ferret "
    "lantern quartz parliament hummingbird vinegar plywood monsoon trombone sapphire "
    "cardboard hedgehog plasma cathedral marmalade pendulum walnut blizzard origami "
    "tarpaulin armadillo basalt courtyard dandelion espresso firefly granite hurricane "
    "iceberg jamboree kerosene labyrinth mahogany nebula obsidian peppermint quokka "
    "rhubarb seaweed tapestry ukulele volcano walrus xylophone yarn zeppelin almanac "
    "barnacle cactus driftwood emerald flamingo gargoyle hammock igloo jigsaw kelp "
    "lagoon mistletoe narwhal otter pinecone quicksand raccoon sandstone thistle "
    "underbrush vineyard windmill yodel zinc accordion bumblebee chandelier dynamo "
    "eclipse fjord glockenspiel harpoon ivy jackal kimono lighthouse marble nutmeg "
    "oasis pomegranate quill ravine sundial trellis urchin viola wombat anchor "
    "bramble citadel domino ember frostbite gondola hibiscus inkwell juniper kettle"
).split()


def canonical() -> tuple[str, str]:
    t = random.choice(TOPICS)
    return t["q"], "exact"


def paraphrase() -> tuple[str, str]:
    t = random.choice(TOPICS)
    return random.choice(t["p"]), "semantic"


def unique() -> tuple[str, str]:
    words = random.sample(VOCAB, random.randint(8, 12))
    return "tell me about " + " ".join(words), "miss"


def pick(p_exact: float = 0.45, p_semantic: float = 0.35) -> tuple[str, str]:
    """Return (prompt, expected_kind) using the configured mix."""
    r = random.random()
    if r < p_exact:
        return canonical()
    if r < p_exact + p_semantic:
        return paraphrase()
    return unique()


def all_canonicals() -> list[str]:
    return [t["q"] for t in TOPICS]
