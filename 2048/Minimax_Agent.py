from Agent import Agent
from GameBoard import GameBoard
import numpy as np


class MinimaxAgent(Agent):
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

        for move in moves:
            child = self.get_child(board, move)

            utility = self.minimax(child, 4, "player1") 

            if utility >= maxUtility:
                maxUtility = utility
                nextDir = move

        return nextDir

    def minimax(self, board, depth, turn):
        if len(board.get_available_moves()) == 0 or depth == 0:
            return self.heuristic_utility(board)
        if turn == "player1":
            bestValue =  -np.inf
            children = []
            for move in board.get_available_moves():
                children.append(self.get_child(board, move))
            for child in children:
                val = self.minimax(child, depth-1, "board")
                bestValue = max(bestValue, val)
                return bestValue
        else:
            bestValue = np.inf
            empty = board.get_available_cells();
            children = []
            
            for pos in empty:
                current_grid2 = board.clone()
                current_grid4 = board.clone()
                
                current_grid2.insert_tile(pos, 2)
                current_grid4.insert_tile(pos, 4)

                children.append(current_grid2)
                children.append(current_grid4)

            for child in children:
                bestValue = min(bestValue, self.minimax(child, depth-1, "player1"))
            return bestValue

    def heuristic_utility(self, board: GameBoard):
        return 0
