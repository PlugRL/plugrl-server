import abc
import dataclasses

@dataclasses.dataclass
class BaseAlgoConfig:
    ...

class BaseAlgorithm(abc.ABC):
    
    @abc.abstractmethod                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    def __init__(self, config: BaseAlgoConfig):
        ...
        
    @abc.abstractmethod
    def infer(self, obs: dict) -> dict:
        ...
        
    @abc.abstractmethod
    def feedback(self, obs: dict, reward: float, terminated: bool, truncated: bool, info: dict) -> None:
        ...