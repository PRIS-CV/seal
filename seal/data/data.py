import typing as t
from torch.utils.data.dataset import Dataset
from torch import Tensor


class Transform:
    
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, input, target):
        raise NotImplementedError("Each transform should implement the __call__ method")

    def __repr__(self) -> str:
        pass


class Pipeline:
    
    def __init__(self, tranforms:t.List[Transform]):
        self.tranforms = tranforms

    def __call__(self, image, bboxes, mask) -> t.Tuple[Tensor, Tensor, Tensor]:
        
        for transform in self.tranforms:
            image, bboxes, mask = transform(image, bboxes, mask)

        return image, bboxes, mask

    def __repr__(self) -> str:
        pipeline_abstract = "\n" + self.__class__.__name__ + ":\n" 
        for transform in self.tranforms:
            pipeline_abstract += transform.__class__.__name__ + "\n"
        return pipeline_abstract


class DataEncoder:
    r"""Data Encoder is aim to encode string of objects or attributes into embedding.
    
    """
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, text: t.Union[t.AnyStr, t.List]) -> t.Union[Tensor, t.List[Tensor]]:
        raise NotImplementedError("Each data encoder should implement the __call__ method.")

    def __repr__(self) -> str:
        return "\n" + self.__class__.__name__ + "\n"
        