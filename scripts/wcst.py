import numpy as np

class WCST:
    
    def __init__(self, batch_size):
        self.colours = ['red','blue','green','yellow']
        self.shapes = ['circle','square','star','cross']
        self.quantities = ['1','2','3','4']
        self.categories = ['C1','C2','C3','C4']
        self.category_feature = np.random.choice([0,1,2])
        self.gen_deck()
        self.batch_size = batch_size

    def gen_deck(self):
        cards = []
        for colour in self.colours:
            for shape in self.shapes:
                for quantity in self.quantities:
                    cards = cards + [(colour, shape, quantity)]
        self.cards = np.array(cards)
        self.card_indices = np.arange(len(cards))

    def context_switch(self):
        self.category_feature = np.random.choice(np.delete([0,1,2],self.category_feature))

    def gen_batch(self):
        batch_size = self.batch_size
        while True:
            prev_feature = self.category_feature
            category_level = np.abs(self.category_feature - 2)+1
            card_partitions = [np.concatenate([np.arange(4**(category_level-1)) + feature_value*(4**(category_level-1))
                              + start for start in np.arange(0,64,int(4**(category_level)))])
                              for feature_value in range(4)]
            category_cards = np.vstack([np.random.choice(card_partition, batch_size, replace=True) for card_partition in card_partitions]).T
            category_cards = category_cards[np.arange(batch_size)[:,np.newaxis], [np.random.permutation(4) for _ in range(batch_size)]]
            category_cards_feature = (category_cards % (4**category_level)) // (4**(category_level-1))
            available_cards = np.delete(np.outer(np.ones((batch_size,1)),self.card_indices).reshape(-1),\
                                       (category_cards+np.arange(batch_size)[:,np.newaxis]*64).reshape(-1)).reshape(batch_size, 60)
            example_cards = available_cards[np.arange(batch_size),np.random.randint(0,60,(batch_size))]
            example_cards_feature = (example_cards % (4**category_level)) // (4**(category_level-1))
            example_labels = np.argmin(np.abs(category_cards_feature - example_cards_feature[:,np.newaxis]), axis=1)
            used_cards = np.hstack([category_cards,example_cards[:,np.newaxis]]).astype(int)
            available_cards = np.delete(np.outer(np.ones((batch_size,1)),self.card_indices).reshape(-1),\
                                       (used_cards+np.arange(batch_size)[:,np.newaxis]*64).reshape(-1)).reshape(batch_size, 59)
            question_cards = available_cards[np.arange(batch_size),np.random.randint(0,59,(batch_size))]
            question_cards_feature = (question_cards % (4**category_level)) // (4**(category_level-1))
            question_labels = np.argmin(np.abs(category_cards_feature - question_cards_feature[:,np.newaxis]),axis=1)
            yield np.hstack([category_cards,example_cards[:,np.newaxis],np.ones((batch_size,1))*68,\
                          example_labels[:,np.newaxis]+64,np.ones((batch_size,1))*69]),\
                   np.hstack([question_cards[:,np.newaxis],np.ones((batch_size,1))*68,question_labels[:,np.newaxis]+64])

    def visualise_batch(self,batch):
        trials = []
        batch = np.hstack(batch)
        for trial_idx in range(batch.shape[0]):
            trial = batch[trial_idx].astype(int)
            trial_cards = []
            for token_idx in trial:
                if token_idx < 64:
                    trial_cards = trial_cards + [self.cards[token_idx]]
                elif token_idx < 68:
                    trial_cards = trial_cards + [self.categories[token_idx-64]]
                elif token_idx == 68:
                    trial_cards = trial_cards + ['SEP']
                elif token_idx == 69:
                    trial_cards = trial_cards + ['EOS']
            trials = trials + [trial_cards]
            print(trial_cards)
        print("Feature for Classification: ", self.category_feature, "\n")
        return trials

wcst = WCST(8)
for i in range(10):
    batch = next(wcst.gen_batch())
    wcst.visualise_batch(batch)
wcst.context_switch()
for i in range(10):
    batch = next(wcst.gen_batch())
    wcst.visualise_batch(batch)
