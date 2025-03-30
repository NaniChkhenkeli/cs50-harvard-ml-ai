import sys
from crossword import Crossword, Variable
from PIL import Image, ImageDraw, ImageFont

class CrosswordCreator:
    def __init__(self, crossword):
        self.crossword = crossword
        self.domains = {
            var: set(self.crossword.words)
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        try:
            cell_size = 100
            cell_border = 3
            interior_size = cell_size - 2 * cell_border
            letters = self.letter_grid(assignment)

            img = Image.new(
                "RGB",
                (self.crossword.width * cell_size,
                 self.crossword.height * cell_size),
                "white"
            )
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arialbd.ttf", 60)
            except:
                font = ImageFont.load_default()

            for i in range(self.crossword.height):
                for j in range(self.crossword.width):
                    rect = [
                        (j * cell_size + cell_border,
                         i * cell_size + cell_border),
                        ((j + 1) * cell_size - cell_border,
                         (i + 1) * cell_size - cell_border)
                    ]
                    
                    if self.crossword.structure[i][j]:
                        draw.rectangle(rect, fill="white")
                        draw.rectangle(rect, outline="black", width=2)
                        
                        if letters[i][j]:
                            text = letters[i][j].upper()
                            _, _, w, h = draw.textbbox((0, 0), text, font=font)
                            draw.text(
                                (rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10),
                                text, fill="black", font=font
                            )
                    else:
                        draw.rectangle(rect, fill="black")

            img.save(filename, "PNG")
            print(f"Saved crossword to {filename}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def solve(self):
        """Enforce node and arc consistency, then solve with backtracking"""
        self.enforce_node_consistency()
        if not self.ac3():
            return None
        return self.backtrack({})

    def enforce_node_consistency(self):
        """Update domains to only include words of correct length"""
        for var in self.domains:
            length = var.length
            self.domains[var] = {
                word for word in self.domains[var]
                if len(word) == length and word in self.crossword.words
            }

    def revise(self, x, y):
        """Make x arc consistent with y"""
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False
        
        i, j = overlap
        x_domain = list(self.domains[x])
        y_domain = self.domains[y]
        
        for word_x in x_domain:
            match_found = False
            for word_y in y_domain:
                if word_x[i] == word_y[j]:
                    match_found = True
                    break
            if not match_found:
                self.domains[x].remove(word_x)
                revised = True
        return revised

    def ac3(self, arcs=None):
        """Enforce arc consistency using AC3 algorithm"""
        queue = []
        if arcs is None:
            for x in self.domains:
                for y in self.crossword.neighbors(x):
                    if y != x:
                        queue.append((x, y))
        else:
            queue = list(arcs)
        
        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x):
                    if z != y:
                        queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        """Check if assignment is complete"""
        return all(var in assignment for var in self.crossword.variables)

    def consistent(self, assignment):
        """Check if assignment is consistent"""
        values = list(assignment.values())
        if len(values) != len(set(values)):
            return False
        
        for var in assignment:
            if var.length != len(assignment[var]):
                return False
            
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap:
                        i, j = overlap
                        if assignment[var][i] != assignment[neighbor][j]:
                            return False
        return True

    def order_domain_values(self, var, assignment):
        """Order values by least constraining value heuristic"""
        def count_eliminations(value):
            count = 0
            for neighbor in self.crossword.neighbors(var):
                if neighbor not in assignment:
                    overlap = self.crossword.overlaps[var, neighbor]
                    if overlap:
                        i, j = overlap
                        for neighbor_value in self.domains[neighbor]:
                            if value[i] != neighbor_value[j]:
                                count += 1
            return count
        
        return sorted(self.domains[var], key=lambda x: count_eliminations(x))

    def select_unassigned_variable(self, assignment):
        """Select unassigned variable using MRV and degree heuristics"""
        unassigned = [var for var in self.crossword.variables if var not in assignment]
        return min(unassigned, key=lambda var: (len(self.domains[var]), -len(self.crossword.neighbors(var))))

    def backtrack(self, assignment):
        """Backtracking search with inference"""
        if self.assignment_complete(assignment):
            return assignment
            
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            
            if self.consistent(new_assignment):
                old_domains = {v: self.domains[v].copy() for v in self.domains}
                
                if self.ac3([(n, var) for n in self.crossword.neighbors(var) if n not in new_assignment]):
                    result = self.backtrack(new_assignment)
                    if result is not None:
                        return result
                
                self.domains = old_domains
                
        return None

def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output.png]")

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else "output.png"

    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    if assignment is None:
        print("No solution exists with the given word list.")
    else:
        creator.print(assignment)
        creator.save(assignment, output)

if __name__ == "__main__":
    main()


# cd C:/Users/Dell/Desktop/hw3/crossword
# python generate.py data/structure1.txt data/words1.txt


