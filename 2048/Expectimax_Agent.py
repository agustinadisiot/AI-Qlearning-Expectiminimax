from Agent import Agent
from GameBoard import GameBoard
import numpy as np


class ExpectimaxAgent(Agent):
    def init(self):
        pass

    def play(self, board:GameBoard):
        depth = 4
        move, _ = self.expectimax(board, depth)
        return move

    def expectimax(self, board, depth = 0):
        moves = board.get_available_moves()
        moves_boards = []

        for m in moves:
            m_board = board.clone()
            m_board.move(m)
            moves_boards.append((m, m_board))

        max_utility = (float('-inf'),0,0,0)
        best_direction = None

        for mb in moves_boards:
            utility = self.chance(mb[1], depth + 1)

            if utility[0] >= max_utility[0]:
                max_utility = utility
                best_direction = mb[0]

        return best_direction, max_utility

    def chance(self, board, depth = 0):
       
        empty_cells = board.get_available_cells()
        n_empty = len(empty_cells)

        if n_empty >= 6 and depth >= 3:
            return self.heuristic_utility(board)

        if n_empty >= 0 and depth >= 5:
            return self.heuristic_utility(board)

        if n_empty == 0:
            _, utility = self.expectimax(board, depth + 1)
            return utility

        possible_tiles = []

        chance_2 = (.9 * (1 / n_empty))
        chance_4 = (.1 * (1 / n_empty))
        
        for empty_cell in empty_cells:
            possible_tiles.append((empty_cell, 2, chance_2))
            possible_tiles.append((empty_cell, 4, chance_4))

        utility_sum = [0, 0, 0, 0]

        for t in possible_tiles:
            t_board = board.clone()
            t_board.insert_tile(t[0], t[1])
            _, utility = self.expectimax(t_board, depth + 1)

            for i in range(4):
                utility_sum[i] += utility[i] * t[2]
        return tuple(utility_sum)

    def heuristic_utility(self, board: GameBoard):
        return (0,0,0,0)
