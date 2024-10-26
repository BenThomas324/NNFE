
from abc import ABC, abstractmethod

class FE_Base(ABC):
    def __init__(self):
        return
    
    @abstractmethod
    def fe_setup(self):
        return
    
    @abstractmethod
    def get_res(self):
        return

    @abstractmethod
    def get_training_data(self):
        return
    
    @abstractmethod
    def get_testing_data(self):
        return

    
    