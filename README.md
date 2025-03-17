../hw0 solution descriptions are provided in the project files. 

## ../hw1/minesweeper ##
project implements ai agent that can play the classic game of minesweeper using propositional logic and fnowledge-based reasoning. 
 player must identify all the mines on a grid without clicking on them. ai uses  logical reasoning to determine which cells are safe to click and which cells are likely to contain mines.

 how works:
 AI starts with no knowledge about the bpard. 
 makes ramdon move. 
 when a safe cell is clicked, ai learns number of neighboring mines and updates its knowledge base. 
 ai uses logical inference to deduce which cells are safe and which are mines:
   if sentence has count of 0, all cells are safe.
   if sentence has count equal to number of cells, all its cells are mines. 
   if one sentence is subset of another, diff can form new sentence. 
 ai wins if identifies all mines. loses if clicks on a mine. 
## sentence class ## 
  known_mines: Returns the set of cells that are definitely mines.
  known_safes: Returns the set of cells that are definitely safe.
  mark_mine: Updates the sentence when a cell is confirmed to be a mine.
  mark_safe: Updates the sentence when a cell is confirmed to be safe.
## minesweeperai class ## 
moves_made: cells that have already been clicked.
mines: cells known to be mines.
safes: cells known to be safe.
knowledge: list of logical sentences representing the AI's knowledge.

key methods in this class are:
add_knowledge: updates the ai's knowledge base with new information about a cell and its neighboring mines.
make_safe_move: returns a safe cell to click.
make_random_move: returns a random cell to click when no safe moves are available.



## ..\hw1\knights ##
this project slves logic puzzle and uses logical sentences to represent statement made by characters and determines who is knight or knave. 

 how works:
 each character (A, B, C) is represented by two symbols: knight (truth-teller) and knave (liar).
 ex: AKnight means "A is a Knight," and AKnave means "A is a Knave."
 statements made by characters are translated into logical sentences using And, Or, Implication, and Biconditional.
 ex: If A says, "I am a Knight," this is represented as:
 biconditional(AKnight, AKnight)
 model_check function checks all possible combinations of truth assignments to determine which symbols are consistent with the knowledge base.

## puzzles ##
puzzle 0:
A says i am both knight and knave. so A must be a Knave because statement is contradiction. 
puzzle 1: 
A says we are both knaves. so if A is knight, statement true, but knights cant lie. => A is knave. -> st false -> B knight. 
puzzle 2: 
A says we are same kind. B says we are different. if A is knight, B must be knight, B's st shold be false, thats inpossible. => A knave, B knight.
puzzle 3:
A says either i am knight or i am knave. => true, A is knight. 
B says i am knave, c is knave. => B's st are contradictory, so B is knave. 
C says A is knight. => true, c is knight. 


for running the code - 

install dependecies: pip install -r requirements.txt
then, run the solver: python puzzle.py (for minesweeper use ## python runner.py ## )


