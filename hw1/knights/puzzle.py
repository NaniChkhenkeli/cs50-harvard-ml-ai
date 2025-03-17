from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
knowledge0 = And(
    AKnave  # A must be a knave because the statement is a contradiction
)

# Puzzle 1
knowledge1 = And(
    Or(AKnight, AKnave),  # A must be either a knight or a knave
    Or(BKnight, BKnave),  # B must be either a knight or a knave
    Not(And(AKnight, AKnave)),  # A cannot be both
    Not(And(BKnight, BKnave)),  # B cannot be both
    Implication(AKnight, And(AKnave, BKnave)),  # If A is a knight, then A and B must both be knaves (impossible)
    Implication(AKnave, Not(And(AKnave, BKnave)))  # If A is a knave, then "We are both knaves" is false, so B is a knight
)

# Puzzle 2
knowledge2 = And(
    Or(AKnight, AKnave),  # A must be either a knight or a knave
    Or(BKnight, BKnave),  # B must be either a knight or a knave
    Not(And(AKnight, AKnave)),  # A cannot be both
    Not(And(BKnight, BKnave)),  # B cannot be both

    # A's statement: "We are the same kind."
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),

    # B's statement: "We are of different kinds."
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight)))
)

# Puzzle 3
knowledge3 = And(
    Or(AKnight, AKnave),  # A must be either a knight or a knave
    Or(BKnight, BKnave),  # B must be either a knight or a knave
    Or(CKnight, CKnave),  # C must be either a knight or a knave
    Not(And(AKnight, AKnave)),  # A cannot be both
    Not(And(BKnight, BKnave)),  # B cannot be both
    Not(And(CKnight, CKnave)),  # C cannot be both

    # A's statement: Either "I am a knight." or "I am a knave."
    Biconditional(AKnight, Or(AKnight, AKnave)),

    # B's statements:
    Implication(BKnight, And(
        Biconditional(AKnight, AKnave),  # If B is a knight, then A said "I am a knave."
        CKnave  # If B is a knight, then C is a knave.
    )),
    Implication(BKnave, Or(
        Not(Biconditional(AKnight, AKnave)),  # If B is a knave, then A did not say "I am a knave."
        Not(CKnave)  # If B is a knave, then C is not a knave.
    )),

    # C's statement: "A is a knight."
    Biconditional(CKnight, AKnight)
)

def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")

if __name__ == "__main__":
    main()