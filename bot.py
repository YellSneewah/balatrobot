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
        self.running = True
        self.balatro_instance = None

        self.sock = None

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

    def skip_or_select_blind(self):
        raise NotImplementedError(
            "Error: Bot.skip_or_select_blind must be implemented."
        )

    def select_cards_from_hand(self):
        raise NotImplementedError(
            "Error: Bot.select_cards_from_hand must be implemented."
        )

    def select_shop_action(self):
        raise NotImplementedError("Error: Bot.select_shop_action must be implemented.")

    def select_booster_action(self):
        raise NotImplementedError(
            "Error: Bot.select_booster_action must be implemented."
        )

    def sell_jokers(self):
        raise NotImplementedError("Error: Bot.sell_jokers must be implemented.")

    def rearrange_jokers(self):
        raise NotImplementedError("Error: Bot.rearrange_jokers must be implemented.")

    def use_or_sell_consumables(self):
        raise NotImplementedError(
            "Error: Bot.use_or_sell_consumables must be implemented."
        )

    def rearrange_consumables(self):
        raise NotImplementedError(
            "Error: Bot.rearrange_consumables must be implemented."
        )

    def rearrange_hand(self):
        raise NotImplementedError("Error: Bot.rearrange_hand must be implemented.")

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

    def sendcmd(self, cmd, **kwargs):
        msg = bytes(cmd, "utf-8")
        self.sock.sendto(msg, self.addr)

    def actionToCmd(self, action):
        result = []

        for x in action:
            if isinstance(x, Actions):
                result.append(x.name)
            elif type(x) is list:
                result.append(",".join([str(y) for y in x]))
            else:
                result.append(str(x))

        return "|".join(result)

    def verifyimplemented(self):
        try:
            self.skip_or_select_blind(self, {})
            # self.select_cards_from_hand(self, {})
            self.select_shop_action(self, {})
            self.select_booster_action(self, {})
            self.sell_jokers(self, {})
            self.rearrange_jokers(self, {})
            self.use_or_sell_consumables(self, {})
            self.rearrange_consumables(self, {})
            self.rearrange_hand(self, {})
        except NotImplementedError as e:
            print(e)
            sys.exit(0)
        except:
            pass

    def random_seed(self):
        # e.g. 1OGB5WO
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))

    def chooseaction(self):
        match self.G["waitingFor"]:
            # case "start_run":
            #     seed = self.seed
            #     if seed is None:
            #         seed = self.random_seed()
            #     return [
            #         Actions.START_RUN,
            #         self.stake,
            #         self.deck,
            #         seed,
            #         self.challenge,
            #     ]
            case "skip_or_select_blind":
                return self.skip_or_select_blind(self, self.G)
            # case "select_cards_from_hand":
            #     return self.select_cards_from_hand(self, self.G)
            case "select_shop_action":
                return self.select_shop_action(self, self.G)
            case "select_booster_action":
                return self.select_booster_action(self, self.G)
            case "sell_jokers":
                return self.sell_jokers(self, self.G)
            case "rearrange_jokers":
                return self.rearrange_jokers(self, self.G)
            case "use_or_sell_consumables":
                return self.use_or_sell_consumables(self, self.G)
            case "rearrange_consumables":
                return self.rearrange_consumables(self, self.G)
            case "rearrange_hand":
                return self.rearrange_hand(self, self.G)

    def one_hot_encode_hand(self, hand):
        ranks = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 
                 '10':8, 'Jack':9, 'Queen':10, 'King':11, 'Ace':12}  # Rank 1-13 mapped to 0-12
        suits = {"Clubs": 0, "Diamonds": 1, "Hearts": 2, "Spades": 3}  # Suit mapped to 0-3
        
        encoded_hand = []
        
        for pos, card in enumerate(hand):
            position_onehot = np.zeros(8)
            position_onehot[pos] = 1  # One-hot encode position (0-7)
            
            rank_onehot = np.zeros(13)
            rank_onehot[ranks[card["value"]]] = 1  # One-hot encode rank (0-12)
            
            suit_onehot = np.zeros(4)
            suit_onehot[suits[card["suit"]]] = 1  # One-hot encode suit (0-3)
            
            encoded_card = np.concatenate([position_onehot, rank_onehot, suit_onehot])
            encoded_hand.append(encoded_card)
        return np.array(encoded_hand, dtype=np.int8)
    
    def calc_reward(self):
        chips = self.G['chips']
        blind_chips = self.G['current_round']["blind_chips"]
        discards = self.G['current_round']['discards_left']
        hands = self.G['current_round']['hands_left']

        if chips < blind_chips:
            return ((chips - blind_chips) * discards + (chips * (hands + 1))) * self.played_hand
        else:
            return chips * (hands + 1) * self.played_hand

        # return chips / blind_chips

    def run(self):
        while self.running:
            if self.sock is None:
                self.verifyimplemented()
                self.state = {}
                self.G = None

                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(1)
                self.sock.connect(self.addr)

            self.sendcmd("HELLO")
            try:
                jsondata = {}
                data = self.sock.recv(65536)
                jsondata = json.loads(data)

                if "response" in jsondata:
                    print(jsondata["response"])
                else:
                    self.G = jsondata
                    if self.G["waitingForAction"]:
                        cache_state(self.G["waitingFor"], self.G)


                        if self.played_hand is not None and "blind_chips" in self.G['current_round'].keys():
                            self.hand_chips = self.G["chips"] - self.prev_chips
                            self.prev = self.G["chips"]

                            self.reward = self.calc_reward()
                        if self.G["waitingFor"] == "select_cards_from_hand":
                            self.starting = False
                            self.new_hand = self.one_hot_encode_hand(self.G["hand"])
                            print(self.new_hand)
                            return self.new_hand, self.reward, self.done, self.info
                        if self.G["waitingFor"] == "start_run" and not self.done and not self.starting:
                            self.done = True
                            return self.new_hand, self.reward, self.done, self.info
                        else:
                            action = self.chooseaction()
                            if action == None:
                                raise ValueError("All actions must return a value!")

                            cmdstr = self.actionToCmd(action)
                            self.sendcmd(cmdstr)
            except socket.error as e:
                print(e)
                print("Socket error, reconnecting...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.settimeout(1)
                self.sock.connect(self.addr)