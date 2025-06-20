#!/usr/bin/python3

import sys
import json
import socket
import time
from enum import Enum
from gamestates import cache_state
import subprocess
import random
import numpy as np
import socket


class State(Enum):
    SELECTING_HAND = 1
    HAND_PLAYED = 2
    DRAW_TO_HAND = 3 
    GAME_OVER = 4
    SHOP = 5
    PLAY_TAROT = 6
    BLIND_SELECT = 7
    ROUND_EVAL = 8
    TAROT_PACK = 9
    PLANET_PACK = 10
    MENU = 11
    TUTORIAL = 12
    SPLASH = 13
    SANDBOX = 14
    SPECTRAL_PACK = 15
    DEMO_CTA = 16
    STANDARD_PACK = 17
    BUFFOON_PACK = 18
    NEW_ROUND = 19


class Actions(Enum):
    SELECT_BLIND = 1
    SKIP_BLIND = 2
    PLAY_HAND = 3
    DISCARD_HAND = 4
    END_SHOP = 5
    REROLL_SHOP = 6
    BUY_CARD = 7
    BUY_VOUCHER = 8
    BUY_BOOSTER = 9
    SELECT_BOOSTER_CARD = 10
    SKIP_BOOSTER_PACK = 11
    SELL_JOKER = 12
    USE_CONSUMABLE = 13
    SELL_CONSUMABLE = 14
    REARRANGE_JOKERS = 15
    REARRANGE_CONSUMABLES = 16
    REARRANGE_HAND = 17
    PASS = 18
    START_RUN = 19
    SEND_GAMESTATE = 20


class Bot:
    def __init__(
        self,
        deck: str,
        stake: int = 1,
        seed: str = None,
        challenge: str = None,
        bot_port: int = 12346,
    ):
        self.G = None
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.challenge = challenge

        self.bot_port = bot_port

        self.addr = ("localhost", self.bot_port)
        self.balatro_instance = None

        self.state = {}
        self.played_hand = None
        self.hand_chips = 0
        self.prev_chips = 0
        self.starting = False

        self.new_hand = []
        self.done = False
        self.truncated = False
        self.reward = 0
        self.info = {}

    # Any Time Actions
    def use_consumable(self, consumable):
        return [Actions.USE_CONSUMABLE, [consumable]]
    
    def rearrange_consumables(self, order):
        return [Actions.REARRANGE_CONSUMABLES, [order]]
    
    def sell_consumable(self, consumable):
        return [Actions.SELL_CONSUMABLE, [consumable]]
    
    def rearrange_jokers(self, order):
        return [Actions.REARRANGE_JOKERS, [order]]
    
    def sell_jokers(self, joker):
        return [Actions.SELL_JOKER, [joker]]

    # Round Actions
    def play_hand(self, cards):
        return [Actions.PLAY_HAND, [cards]]
    
    def discard_hand(self, cards):
        return [Actions.DISCARD_HAND, [cards]]
    
    def rearrange_hand(self, order):
        return [Actions.REARRANGE_HAND, [order]]

    def next_round(self):
        return [Actions.END_SHOP]

    # Shop Actions
    def next_round(self):
        return [Actions.END_SHOP]

    def reroll_shop(self):
        return [Actions.REROLL_SHOP]
    
    def buy_card(self, card):
        return [Actions.BUY_CARD, [card]]
    
    def buy_and_use_card(self, card):
        return [Actions.BUY_CARD, [card], True]
    
    def buy_voucher(self, voucher_slot):
        return [Actions.BUY_VOUCHER, [voucher_slot]]
    
    def buy_booster(self, booster):
        return [Actions.BUY_BOOSTER, [booster]] 
    
    # Booster Actions
    def skip_booster_pack(self):
        return [Actions.SKIP_BOOSTER_PACK]
    
    def use_booster_pack(self, booster_slot, cards):
        return [Actions.SELECT_BOOSTER_CARD, [booster_slot, cards]]

    # Blind Actions
    def select_blind(self):
        return [Actions.SELECT_BLIND]
    
    def skip_blind(self):
        return [Actions.SKIP_BLIND]
    
    

    def start_balatro_instance(self):
        balatro_exec_path = (
            r"D:\Program Files\Steam\steamapps\common\Balatro\Balatro.exe"
        )
        self.balatro_instance = subprocess.Popen(
            [balatro_exec_path, str(self.bot_port)]
        )

    def stop_balatro_instance(self):
        if self.balatro_instance:
            self.balatro_instance.kill()

    def send_cmd(self, cmd, **kwargs):
        msg = bytes(cmd, "utf-8")
        self.sock.sendto(msg, self.addr)

    def action_to_cmd(self, action):
        result = []

        for x in action:
            if isinstance(x, Actions):
                result.append(x.name)
            elif type(x) is list:
                result.append(",".join([str(y) for y in x]))
            else:
                result.append(str(x))

        return "|".join(result)


    def random_seed(self):
        # e.g. 1OGB5WO
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))


    # def one_hot_encode_hand(self, hand):
    #     ranks = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 
    #              '10':8, 'Jack':9, 'Queen':10, 'King':11, 'Ace':12}  # Rank 1-13 mapped to 0-12
    #     suits = {"Clubs": 0, "Diamonds": 1, "Hearts": 2, "Spades": 3}  # Suit mapped to 0-3
        
    #     encoded_hand = []
        
    #     for pos, card in enumerate(hand):
    #         position_onehot = np.zeros(8)
    #         position_onehot[pos] = 1  # One-hot encode position (0-7)
            
    #         rank_onehot = np.zeros(13)
    #         rank_onehot[ranks[card["value"]]] = 1  # One-hot encode rank (0-12)
            
    #         suit_onehot = np.zeros(4)
    #         suit_onehot[suits[card["suit"]]] = 1  # One-hot encode suit (0-3)
            
    #         encoded_card = np.concatenate([position_onehot, rank_onehot, suit_onehot])
    #         encoded_hand.append(encoded_card)
    #     return np.array(encoded_hand, dtype=np.int8)
    
    def calculate_reward(self, state):
        chips = state['chips']

        return chips

    def get_state(self):
        self.send_cmd("HELLO")
        while True:
            try:
                jsondata = {}
                data = self.sock.recv(65536)
                jsondata = json.loads(data)
                if "response" in jsondata:
                    print(jsondata["response"])
                else:
                    self.G = jsondata
                    return self.G
            except socket.error as e:
                print(e)
                print("Socket error, reconnecting...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(1)
                self.sock.connect(self.addr)


    # def run(self):
    #     while self.running:
    #         self.sendcmd("HELLO")
    #         try:
    #             jsondata = {}
    #             data = self.sock.recv(65536)
    #             jsondata = json.loads(data)

    #             if "response" in jsondata:
    #                 print(jsondata["response"])
    #             else:
    #                 self.G = jsondata
    #                 if self.G["waitingForAction"]:
    #                     cache_state(self.G["waitingFor"], self.G)

    #                     if self.played_hand is not None and "blind_chips" in self.G['current_round'].keys():
    #                         self.hand_chips = self.G["chips"] - self.prev_chips
    #                         self.prev = self.G["chips"]

    #                         self.reward = self.calc_reward()
    #                     if self.G["waitingFor"] == "select_cards_from_hand":
    #                         self.starting = False
    #                         self.new_hand = self.one_hot_encode_hand(self.G["hand"])
    #                         print(self.new_hand)
    #                         return self.new_hand, self.reward, self.done, self.info
    #                     if self.G["waitingFor"] == "start_run" and not self.done and not self.starting:
    #                         self.done = True
    #                         return self.new_hand, self.reward, self.done, self.info
    #                     else:
    #                         action = self.chooseaction()
    #                         if action == None:
    #                             raise ValueError("All actions must return a value!")

                            
    #                         self.sendcmd(action)
    #         except socket.error as e:
    #             print(e)
    #             print("Socket error, reconnecting...")
    #             self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #             self.sock.settimeout(1)
    #             self.sock.connect(self.addr)
