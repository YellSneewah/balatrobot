import gymnasium.spaces as spaces
import gymnasium as gym
from rl_bot import Bot, Actions, State
from stable_baselines3.common.env_checker import check_env

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

import socket
import numpy as np

def mask_fn(env: gym.Env):
    return env.valid_action_mask()


class BalatroEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, bot_init, max_consumables, max_jokers, max_handsize, max_vouchers=1):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Change action_space to a Dict of Discrete spaces


        self.max_consumables = max_consumables
        self.max_jokers = max_jokers
        self.max_handsize = max_handsize
        self.max_vouchers = max_vouchers
        self.max_shop_cards = 4

    


        self.action_vector = [
            6,                                # any_time_action (# No Action, Use Consumable, Rearrange Consumable, Sell Consumable, Rearrange Joker, Sell Joker)
            self.max_consumables,             # selected_consumable
            self.max_consumables,             # new_consumable_index
            self.max_jokers,                  # selected_joker
            self.max_jokers,                  # new_joker_index
            3,                                # round_action (Play, Discard, Rearrange Card)
            self.max_handsize,                # selected_card_1
            self.max_handsize + 1,            # selected_card_2
            self.max_handsize + 1,            # selected_card_3
            self.max_handsize + 1,            # selected_card_4
            self.max_handsize + 1,            # selected_card_5
            self.max_handsize,                # selected_card_rearrange
            self.max_handsize,                # new_card_index
            7,                                # shop_action (Next Round, Reroll, Buy Card, Buy & Use Card, Buy Voucher, Buy Booster 1, Buy Booster 2)
            self.max_shop_cards,              # shop_slot
            self.max_vouchers,                # voucher_slot
            6,                                # booster_action (Skip, Use 1, Use 2, Use 3, Use 4, Use 5)
            2,                                # blind_select (Skip, Select)
        ]

        self.action_space = spaces.MultiDiscrete(self.action_vector)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Dict(spaces = {
            "state": spaces.Discrete(len(State)),
            "dollars": spaces.Box(low=0, high=1000, shape=(1,), dtype = np.int32),
            "score": spaces.Box(low=0, high=1000000, shape=(1,), dtype = np.int32)
        })

        
        self.bot = bot_init()

        self.deck = "Red Deck"
        self.stake = 1
        self.seed = self.bot.random_seed()
        self.challenge = None

        # Variables
        self.state = self.bot.get_state()

        

    def step(self, action):
        action = unflatten_action(action)
        match action["any_time_action"]:
            case 0: 
                # No Action
                pass

            case 1: 
                # Use Consumable
                consumable = action["selected_consumable"]
                if consumable < 0:
                    raise ValueError("Invalid consumable selected")
                if consumable >= self.max_consumables:
                    raise ValueError("Selected consumable out of range")
                # Implement logic to use the consumable
                selected_action = self.bot.use_consumable(consumable)

            case 2:
                # Rearrange Consumables
                consumable = action["selected_consumable"]
                new_index = action["new_consumable_index"]
                num_consumables = len(self.state["consumables"])
                selected_action = self.bot.rearrange_consumables(num_consumables, consumable, new_index)

            case 3: 
                # Sell Consumable
                consumable = action["selected_consumable"]
                if consumable < 0:
                    raise ValueError("Invalid consumable selected")
                if consumable >= self.max_consumables:
                    raise ValueError("Selected consumable out of range")
                # Implement logic to sell the consumable
                selected_action = self.bot.sell_consumable(consumable)

            case 4:
                # Rearrange Joker
                joker = action["selected_joker"]
                placement = action["joker_placement"]
                if joker < 0:
                    raise ValueError("Invalid joker selected")
                if joker >= self.max_jokers:
                    raise ValueError("Selected joker out of range")
                # Implement logic to rearrange the joker
                num_jokers = len(self.state["jokers"])
                selected_action = self.bot.rearrange_joker(num_jokers, joker, placement)

            case 5:
                # Sell Joker
                joker = action["selected_joker"]
                if joker < 0:
                    raise ValueError("Invalid joker selected") 
                if joker >= self.max_jokers:
                    raise ValueError("Selected joker out of range")
                # Implement logic to sell the joker
                selected_action = self.bot.sell_joker(joker)

        match action["round_action"]:
            case 0:
                # Play Hand
                cards = list(set([action["selected_card_1"], action["selected_card_2"], action["selected_card_3"], action["selected_card_4"], action["selected_card_5"]]))
                if (self.max_handsize + 1) in cards:
                    cards.pop()
                    raise ValueError("Invalid hand selected")
                selected_action = self.bot.play_hand(cards)

            case 1:
                # Discard Hand
                cards = list(set([action["selected_card_1"], action["selected_card_2"], action["selected_card_3"], action["selected_card_4"], action["selected_card_5"]]))
                if (self.max_handsize + 1) in cards:
                    cards.pop()
                    raise ValueError("Invalid hand selected")
                selected_action = self.bot.discard_hand(cards)

            case 2:
                # Rearrange Hand
                num_in_hand = len(self.state["hand"])
                card = action["selected_card_index"]
                new_index = action["new_card_index"]
                selected_action = self.bot.rearrange_hand(num_in_hand, card, new_index)

        match action["shop_action"]:
            case 0:
                # Next Round
                selected_action = self.bot.next_round()

            case 1:
                # Reroll Shop
                selected_action = self.bot.reroll_shop()

            case 2:
                # Buy Card
                card_slot = action["shop_slot"]
                if card_slot < 0 or card_slot >= 4:
                    raise ValueError("Invalid shop slot selected")
                selected_action = self.bot.buy_card(card_slot)

            case 3:
                # Buy & Use Card
                card_slot = action["shop_slot"]
                if card_slot < 0 or card_slot >= 4:
                    raise ValueError("Invalid shop slot selected")
                selected_action = self.bot.buy_and_use_card(card_slot)

            case 4:
                # Buy Voucher
                voucher_slot = action["voucher_slot"]
                if voucher_slot > 0:
                    voucher_slot = 0
                selected_action = self.bot.buy_voucher(voucher_slot)

            case 5:
                # Buy Booster 1
                selected_action = self.bot.buy_booster(1)

            case 6:
                # Buy Booster 2
                selected_action = self.bot.buy_booster(2)

        match action["booster_action"]:
            case 0:
                # Skip Booster
                selected_action = self.bot.skip_booster_pack()

            case _:
                # Use Booster
                cards = list(set([action["selected_card_1"], action["selected_card_2"], action["selected_card_3"], action["selected_card_4"], action["selected_card_5"]]))

                if (self.max_handsize + 1) in cards:
                    cards.pop()
                    raise ValueError("Invalid booster selected")
                selected_action = self.bot.use_booster_pack(action["booster_action"], cards)

        match action["blind_select"]:
            case 0:
                # Play Blind
                selected_action = self.bot.select_blind()
            case 1:
                # Skip Blind
                selected_action = self.bot.skip_blind()

        self.bot.send_cmd(self.bot.action_to_cmd(selected_action))

        waiting_for_action = False
        while not waiting_for_action:
            self.state = self.bot.get_state()
            if self.state["waitingForAction"]:
                waiting_for_action = True
        reward = self.bot.calculate_reward(self.state)
        terminated = self.state["game_over"]
        truncated = False  # Balatro does not have a time limit, so this is always False
        info = {
            "reward": reward,
            "state": self.state,
            "seed": self.seed
        }
        
        observation = state_to_obs(self.state)
        return observation, reward, terminated, truncated, info
    
    def valid_action_mask(self):
        actions = {
            "any_time_action": [False] * 6,                                 # No Action, Use Consumable, Rearrange Consumable, Sell Consumable, Rearrange Joker, Sell Joker
            "selected_consumable": [False] * self.max_consumables,    
            "new_consumable_index": [False] * self.max_consumables,         # New order for the consumables
            "selected_joker": [False] * self.max_jokers,   
            "new_joker_index": [False] * self.max_jokers,
            "round_action": [False] * 3,                                     # Play, Discard, Rearrange Cards
            "selected_card_1": [False] * self.max_handsize,    
            "selected_card_2": [False] * (self.max_handsize + 1),             # +1 for the case of no card selected
            "selected_card_3": [False] * (self.max_handsize + 1),             # +1 for the case of no card selected
            "selected_card_4": [False] * (self.max_handsize + 1),             # +1 for the case of no card selected
            "selected_card_5": [False] * (self.max_handsize + 1),             # +1 for the case of no card selected
            "selected_card_index": [False] * self.max_handsize,
            "new_card_index": [False] * self.max_handsize,                  # Order of the cards in the hand
            "shop_action": [False] * 7,                                     # Next Round, Reroll, Buy Card, Buy & Use Card, Buy Voucher, Buy Booster 1, Buy Booster 2
            "shop_slot": [False] * 4,
            "voucher_slot": [False] * self.max_vouchers,                    # Useful when voucher skip used
            "booster_action": [False] * 6,                                  # Skip, Use 1, Use 2, Use 3, Use 4, Use 5
            "blind_select": [False] * 2
        }
        G = self.state
        state = G["state"]
        shop = G["shop"]
        jokers = G["jokers"]
        hand = G["hand"]
        consumables = G["consumables"]

        # All Time Action | No Action, Use Consumable, Rearrange Consumable, Sell Consumable, Rearrange Joker, Sell Joker
        if state == (State.SELECTING_HAND or State.HAND_PLAYED or State.DRAW_TO_HAND or State.SHOP or State.PLAY_TAROT or State.BLIND_SELECT or State.ROUND_EVAL or State.PLANET_PACK):

            # Can always do no action
            can_no_action = True

            # Use and sell consumables if you have them
            if len(consumables) > 0:
                can_use_consumable = True
                can_sell_consumable = True
                actions["selected_consumable"] = self.true_list(len(consumables), self.max_consumables)
            # Can rearrange consumables if you have 2 or more
            if len(consumables) > 1:
                can_rearrange_consumables = True
                actions["new_consumable_index"] = self.true_list(len(consumables), self.max_consumables)
            
            # Can sell jokers if you have them
            if len(jokers) > 0:
                for joker in jokers:
                    if not joker["eternal"]:

                        can_sell_joker = True
                actions["selected_joker"] = self.true_list(len(jokers), self.max_jokers)
            # Can rearrange jokers if you have 2 or more
            if len(jokers) > 1:
                can_rearrange_jokers = True
                actions["new_joker_index"] = self.true_list(len(jokers), self.max_jokers)
            
            actions["any_time_action"] = [can_no_action, can_use_consumable, can_rearrange_consumables, can_sell_consumable, can_rearrange_jokers, can_sell_joker]
            

        # In Round Actions
        if state == State.SELECTING_HAND:
            # Can always play hand
            can_play = True

            # Each selection has num_cards in hand choices, duplicates are ignored.
            
            actions["selected_card_1"] = self.true_list(len(hand), self.max_handsize)
            for i in range(2, 6):
                actions["selected_card_" + i] = self.true_list(len(hand), self.max_handsize + 1)
                actions["selected_card_" + i][-1] = True
            


            # Discard hand
            if G["remaining_discards"] > 0:
                can_discard = True

            # Can Rearrange Hand
            if len(hand) > 1:
                can_rearrange_hand = True

            # Can only select cards in hand
            actions["selected_card_rearrange"] = self.true_list(len(hand), self.max_handsize)
            actions["new_card_index"] = self.true_list(len(hand), self.max_handsize)
            actions["round_action"] = [can_play, can_discard, can_rearrange_hand]
        
        # Shop Actions    
        if state == State.SHOP:

            # Next Round
            next_round = True

            # Reroll
            if self.can_afford(shop["reroll_cost"]):
                can_reroll = True

            # Buy and Buy and Use Card
            can_buy_cards = [False] * self.max_shop_cards
            for card, i in enumerate(shop["cards"]):
                if self.can_afford(card, True):
                    if (card["set"] == "Joker"):
                        if (len(jokers) < G["max_jokers"]):
                            can_buy_cards[i] = True
                    elif (card["set" == "Tarot"]):
                        if (len("consumables") < G["max_consumables"]):
                            can_buy_cards[i] = True
            actions["shop_slot"] = can_buy_cards
            if True in can_buy_cards:
                can_buy_cards_bool = True
                

            # Buy Vouchers
            if self.can_afford(shop["vouchers"][0], True):
                can_buy_voucher = True

            # Buy Booster 1
            if self.can_afford(shop["boosters"][0], True):
                can_buy_booster_1 = True

            # Buy Booster 2
            if self.can_afford(shop["boosters"][1], True):
                can_buy_booster_2 = True

            # Next Round, Reroll, Buy Card, Buy & Use Card, Buy Voucher, Buy Booster 1, Buy Booster 2
            actions["shop_action"] = [next_round, can_reroll, can_buy_cards_bool, can_buy_cards_bool, can_buy_voucher, can_buy_booster_1, can_buy_booster_2]

        # Blind Options
        if state == State.BLIND_SELECT:
            # Can always play
            can_select_blind = True

            # Can skip big and small blind
            if G["blind"] == ("small_blind" or "big_blind"):
                can_skip_blind = True
            else:
                can_skip_blind = False
            actions["blind_select"] = [can_select_blind, can_skip_blind]


        print(flatten_actions(actions))
        return flatten_actions(actions)
    


    

    def reset(self, seed=None, options=None):
        self.bot.send_cmd(self.bot.action_to_cmd([Actions.START_RUN, "Red Deck", 1, self.bot.random_seed()]))
        self.state = self.bot.get_state()
        if self.state is None:
            raise RuntimeError("Failed to get initial state from Balatro")
        print("Resetting... Ended at state:", self.state)
        observation = state_to_obs(self.state)
    
        info = {
            "reward": 0,
            "state": self.state
        }
        return observation, info

    def render(self):
        ...

    def close(self):
        self.bot.stop_balatro_instance()

    def can_afford(self: gym.Env, item: any, is_card: bool) -> bool:
        """
        Returns a bool based on whether the item can be bought. Accounts for credit card.
        """
        money = self.state["dollars"]
        bankrupt_at = self.state["bankrupt_at"]
        if is_card:
            return (money - item["cost"] >= bankrupt_at)
        else:
            return (money - item >= bankrupt_at)
    

    def true_list(self, num: int, list: list):
        bool_list = [True] * num
        bool_list.append([False] * (list - num))
        return bool_list 

def flatten_actions(actions: dict):
    #Converts a dictionary of lists into a list of lists.
    output_list = []
    for value in actions.values():
      output_list.append(value)
    return output_list

def unflatten_action(vec):
    it = iter(vec)
    return {
        "any_time_action":      next(it),
        "selected_consumable":  next(it),
        "new_consumable_index": next(it),
        "selected_joker":       next(it),
        "new_joker_index":      next(it),
        "round_action":         next(it),
        "selected_card_1":      next(it),
        "selected_card_2":      next(it),
        "selected_card_3":      next(it),
        "selected_card_4":      next(it),
        "selected_card_5":      next(it),
        "selected_card_index":  next(it),
        "new_card_index":       next(it),
        "shop_action":          next(it),
        "shop_slot":            next(it),
        "voucher_slot":         next(it),
        "booster_action":       next(it),
        "blind_select":         next(it),
    }

def state_to_obs(state):
    obs = {}
    obs["state"] = state["state"]
    obs["dollars"] = state["dollars"]
    obs["score"] = state["handscores"]
    return obs

def init_bot():
    # mybot = Bot(deck="Plasma Deck", stake=1, seed="1OGB5WO")
    mybot = Bot(deck = "Red Deck", stake = 1)
    mybot.start_balatro_instance()

    mybot.running = False
    while mybot.sock == None:
        mybot.state = {}
        mybot.G = None

        mybot.running = True
        mybot.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        mybot.sock.settimeout(1)
        mybot.sock.connect(mybot.addr)
        mybot.send_cmd("HELLO")

    return mybot

# test_dict = {
#     "gaming":[True, True, True],
#     "gaming 2":[False, False, False],
#     "test":[False],
#     "embedded_dict": {
#         "embed 1": [True, True],
#         "embed 2": [False]
#     }
# }

# test_dict["test"] = True

# print(test_dict.values())


env = BalatroEnv(init_bot, max_consumables=5, max_jokers=10, max_handsize=12, max_vouchers=1)
env = ActionMasker(env, mask_fn)
env = gym.wrappers.FlattenObservation(env)


model = MaskablePPO(MaskableActorCriticPolicy, env)
model.learn(total_timesteps=10000)