from Agent import Agent
from GameBoard import GameBoard
import numpy as np


class ExpectimaxAgent(Agent):
    def init(self):
        pass

    def get_child(self, board, move):
        temp = board.clone()
        temp.move(move)
        return temp

    def play(self, board:GameBoard):
        moves = board.get_available_moves()
        maxUtility = -np.inf
        nextDir = -1
        depth = 3


        for move in moves:
            child = self.get_child(board, move)

            utility = self.expectimax(child, depth, "board") 

            if utility >= maxUtility:
                maxUtility = utility
                nextDir = move

        return nextDir

    def check_win(self, board: GameBoard):
        return board.get_max_tile() >= 2048

    def expectimax(self, board, depth, turn):
        if self.check_win(board):
            return np.inf 
        if len(board.get_available_moves()) == 0:
            return -np.inf
        if depth == 0:
            return self.heuristic_utility(board)[0]
        if turn == "player1":
            bestValue =  -np.inf
            children = []
            for move in board.get_available_moves():
                children.append(self.get_child(board, move))
            for child in children:
                val = self.expectimax(child, depth-1, "board")
                bestValue = max(bestValue, val)
            return bestValue
        else:
            #bestValue = np.inf
            empty = board.get_available_cells();

            chance_2 = (.9 * (1 / len(empty)))
            chance_4 = (.1 * (1 / len(empty)))
            children = []
            total = 0
            for pos in empty:
                current_grid2 = board.clone()
                current_grid4 = board.clone()
                
                current_grid2.insert_tile(pos, 2)
                current_grid4.insert_tile(pos, 4)

                val2 = self.expectimax(current_grid2, depth-1, "player1")
                val4 =self.expectimax(current_grid4, depth-1, "player1")
                total += val2 * chance_2 + val4 * chance_4

            return total


    def heuristic_utility(self, board: GameBoard):

        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        #combinacion de todas las heuristicas recomendadas: smoothness (smooth) + valor del tablero (valorT) + vacios (empty)
        grid = board.grid

        utility = 0
        smoothness = 0

        s_grid = np.sqrt(grid) #Aplicar la raiz cuadrada al tablero

        #sumar cada casilla  con la de su derecha y la de abajo y luego multiplicar por -1 
        smoothness += np.sum(np.abs(s_grid[::,0] - s_grid[::,1]))
        smoothness += np.sum(np.abs(s_grid[::,1] - s_grid[::,2]))
        smoothness += np.sum(np.abs(s_grid[::,2] - s_grid[::,3]))
        smoothness += np.sum(np.abs(s_grid[0,::] - s_grid[1,::]))
        smoothness += np.sum(np.abs(s_grid[1,::] - s_grid[2,::]))
        smoothness += np.sum(np.abs(s_grid[2,::] - s_grid[3,::]))
        
        # Elevar este resultado a un smoothness_weight a determinar
        smoothness_weights = 3
        smooth = -1*smoothness ** smoothness_weights 

        #Multiplicar por un empty_weight (recomendable en el orden de las decenas de miles)
        empty_weights = 100000
        empty = n_empty * empty_weights

        #Elevar el tablero al cuadrado y sumar todos los valores que se encuentran en el tablero
        valorT = np.sum(np.power(grid, 2)) 

        #sumar todas las utilidades
        utility += valorT
        utility += empty
        utility += smooth

        return (utility, empty, smooth, valorT)
