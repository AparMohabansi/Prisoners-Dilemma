from ..Agent import Agent
import torch

class HumanPlayer(Agent):
    def __init__(self):
        self.name = input("Enter your name: ")

    def next_move(self, agent_moves, opponent_moves):
        if len(agent_moves) == 0:
            print(f"Welcome to Iterated Prisoners' Dilemma, {self.name}!")
        else:
            opponents_action = "cooperated" if opponent_moves[-1] else "defected"
            print(f"Your opponent {opponents_action}")
        while True:
            action = input("Cooperate, Defect, or view past actions? (c / d / V)  ")
            if action not in ["c", "d", "V"]:
                print(f"Incorrect input. Your input: {action}")
            else:
                if action == "V":
                    print(f"Your moves: \n{agent_moves}")
                    print(f"Opponent moves: \n{opponent_moves}")
                else:
                    if action == 'c':
                        return torch.tensor([1])
                    else:
                        return torch.tensor([0])