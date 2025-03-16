from typing import List, Tuple, Dict, Literal
from Training.Bot import Bot
from Training.Agent import Agent


# The user class, this is used to represent the user and its actions
class User:
    def __init__(self, name: str):
        self.name = name
        self.score = 0

    def choose_action(self):
        # Placeholder for decision logic
        action = input("Cooperate or Defect? ")
        while action != "Cooperate" or action != "Defect":
            print("Incorrect input.")
            action = input("Cooperate or Defect? ")
            
        return action

    def update_score(self, points: int):
        self.score += points

    

class Game:
    def __init__(self, player1, player2, rounds=5):
        self.player1 = player1
        self.player2 = player2
        self.rounds = rounds

    def play_round(self):
        player1_action = self.player1.predict()
        player2_action = self.player2.choose_action()

        if player1_action == "Cooperate":
            if player2_action == "Cooperate":
                self.player1.update_score(3)
                self.player2.update_score(3)
            else:
                self.player2.update_score(5)
        else:
            if player2_action == "Cooperate":
                self.player1.update_score(5)
            else:
                self.player1.update_score(1)
                self.player2.update_score(1)

    def play_game(self):
        for _ in range(self.rounds):
            self.play_round()

def main():
    prisoner1 = Bot("Prisoner 1")
    UserName = input("Enter your name: ")
    prisoner2 = User(UserName)

    game = Game(prisoner1, prisoner2)

    rounds = 5
    for _ in range(rounds):
        game.play_round()

    print(f"{prisoner1.name} Score: {prisoner1.score}")
    print(f"{prisoner2.name} Score: {prisoner2.score}")

if __name__ == "__main__":
    main()