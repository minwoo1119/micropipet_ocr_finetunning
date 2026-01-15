class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        """
        patience  : 개선 없이 허용할 epoch 수
        min_delta : 성능 개선으로 인정할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
