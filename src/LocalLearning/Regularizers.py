from abc import ABC, abstractmethod
import functools

import torch
from torch import Tensor

from .Trainers import Trainer

'''
Regularizers are implemented as Decorators modifying Trainer in Trainers.py
'''

class Regularizer(ABC):
    '''
    abstract Regularizer Decorator class:e
    '''

    def __init__(self, alpha_reg=0.0, **kwargs):

        def decorateTrainer(TrainerCls: Trainer):
            assert issubclass(TrainerCls, Trainer)

            @functools.wraps(TrainerCls, updated=())
            class Wrapper(TrainerCls):

                # add Regularizer method and coupling constant to 
                # Trainer constructor
                def __init__(wref, *args, **kwargs):
                    # wref - wrapper object reference
                    super(Wrapper, wref).__init__(*args, **kwargs)

                    # type(self) -> reference to class of current object
                    # Register Regularizer method
                    setattr(wref, type(self).__name__, self.reg)
                    setattr(wref, type(self).__name__ + "_eval", self.reg_eval)
                    # Register Regularizer coupling parameter
                    setattr(wref, "alpha_" + type(self).__name__, alpha_reg)

                    # register new logging registers
                    wref.log[type(self).__name__ + "_loss"] = []
                    wref.log["eval_" + type(self).__name__ + "_score"] = []

                # modify Trainer's interface functions
                def _batch_loss(
                        wref, 
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        outputs: torch.Tensor,
                        hidden_repr: torch.Tensor,
                        ) -> torch.Tensor:
                    loss = super(Wrapper, wref)._batch_loss(features, labels, outputs, hidden_repr)
                    reg = getattr(wref, type(self).__name__)
                    alpha_reg = getattr(wref, "alpha_" + type(self).__name__)
                    loss += alpha_reg*reg(wref, features, labels, outputs, hidden_repr)
                    return loss

                def _batch_eval(
                        wref,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        predictions: torch.Tensor,
                        hidden_repr: torch.Tensor,
                        ) -> float:
                    frequency = super(Wrapper, wref)._batch_eval(features, labels, predictions, hidden_repr)
                    reg_eval = getattr(wref, type(self).__name__ + "_eval")
                    reg_eval(wref, features, labels, predictions, hidden_repr)
                    return frequency

                def _batch_preprocessing(
                        wref,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        ) -> tuple:
                    features, labels = super(Wrapper, wref)._batch_preprocessing(features, labels)
                    features, labels = self._batch_preprocessing(wref, features, labels)
                    return (features, labels)

                def _batch_postprocessing(
                        wref,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        ) -> None:
                    super(Wrapper, wref)._batch_postprocessing(features, labels)
                    self._batch_postprocessing(wref, features, labels)


                def _epoch_preprocessing_train(
                        wref,
                        ) -> None:
                    super(Wrapper, wref)._epoch_preprocessing_train()
                    self._epoch_preprocessing_train(wref)

                def _epoch_postprocessing_train(
                        wref,
                        ) -> None:
                    super(Wrapper, wref)._epoch_postprocessing_train()
                    self._epoch_postprocessing_train(wref)

                def _epoch_preprocessing_eval(
                        wref,
                        ) -> None:
                    super(Wrapper, wref)._epoch_preprocessing_eval()
                    self._epoch_preprocessing_eval(wref)
                    
                def _epoch_postprocessing_eval(
                        wref,
                        ) -> None:
                    super(Wrapper, wref)._epoch_postprocessing_eval()
                    self._epoch_postprocessing_eval(wref)

            return Wrapper
        self.decorateTrainer = decorateTrainer

    def __call__(self, *args, **kwargs):
        return self.decorateTrainer(*args, **kwargs)
    
    @classmethod
    @abstractmethod
    def reg(
            cls,
            obj_ref,
            features: torch.Tensor,
            labels: torch.Tensor,
            outputs: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> torch.Tensor:
        pass

    @classmethod
    def reg_eval(
            cls,
            obj_ref,
            features: torch.Tensor,
            labels: torch.Tensor,
            predictions: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> None:
        return None

    @classmethod
    def _epoch_preprocessing_train(
            cls,
            obj_ref,
            ) -> None:
        return None

    @classmethod
    def _epoch_postprocessing_train(
            cls,
            obj_ref,
            ) -> None:
        return None

    @classmethod
    def _epoch_preprocessing_eval(
            cls,
            obj_ref,
            ) -> None:
        return None

    @classmethod
    def _epoch_postprocessing_eval(
            cls,
            obj_ref, 
            ) -> None:
        return None
    
    @classmethod
    def _batch_preprocessing(
            cls,
            obj_ref,
            features: torch.Tensor,
            labels: torch.Tensor,
            ) -> tuple:
        return (features, labels)

    @classmethod
    def _batch_postprocessing(
            cls,
            obj_ref,
            features: torch.Tensor,
            labels: torch.Tensor,
            ) -> None:
        return None


class JFReg(Regularizer):
    '''
    Implementation from Hoffman et al.
    '''
    no_projections = 1

    def __init__(self, alpha_JF=0.0, n=1):
        super(JFReg, self).__init__(alpha_reg=alpha_JF)
        no_projections = n
        assert (no_projections == -1) or (no_projections > 0)
        
        if no_projections == -1:
            #setattr(self, "_JF", self._exact_JFReg)
            JFReg._JF = self._exact_JFReg
        else:
            #setattr(self, "_JF", self._projected_JFReg)
            JFReg._JF = self._projected_JFReg
        
    def _exact_JFReg(
            self, 
            features: torch.Tensor, 
            labels: torch.Tensor,
            outputs: torch.Tensor,
            hidden_repr: torch.Tensor,
            ):
        '''
        scales terribly and is basically not usable with hidden layer
        '''
        JF = 0
        for i in range(hidden_repr.size(dim=-1)):
            # define projection tensors
            # in the exact case, these are the canonical basis vectors 
            # of the hidden layer representation
            v = torch.zeros_like(hidden_repr)
            v[:, i] = 1
            J, = torch.autograd.grad(hidden_repr, features, 
                                     grad_outputs=v,
                                     retain_graph=True,
                                     create_graph=True,
                                    )
            JF += J.norm(p='fro')**2
        return JF
    
    def _projected_JFReg(
            self, 
            features: torch.Tensor, 
            labels: torch.Tensor,
            outputs: torch.Tensor,
            hidden_repr: torch.Tensor,
            ):
        # hidden_repr: Tensor, size=(BATCH_SIZE, HIDDEN_SIZE) - model representation
        # features: Tensor, size=(BATCH_SIZE, INPUT_SIZE) - model input
        BATCH_SIZE, HIDDEN_SIZE = hidden_repr.size()
        JF = 0
        for _ in range(type(self).no_projections):
            # sample random vectors on unit sphere
            v = torch.normal(mean=0.0, std=torch.ones_like(hidden_repr)).to(device=hidden_repr.device)
            #v = torch.ones_like(y).to(y.device)
            v /= v.norm(p=2, dim=-1)[..., None] 
            
            hidden_repr = hidden_repr.flatten()
            Jv, = torch.autograd.grad(hidden_repr, features,
                                     grad_outputs=v.flatten(),
                                     retain_graph=True,
                                     create_graph=True,
                                    )
            JF += HIDDEN_SIZE*Jv.norm(p=2)**2 / (self.no_projections*BATCH_SIZE)
        return JF

    @classmethod
    def reg(
        cls,
        obj_ref,
        features: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        hidden_repr: torch.Tensor,
        ) -> torch.Tensor:
        
        jf_loss = cls._JF(features, labels, outputs, hidden_repr)
        obj_ref.cumm_jf_loss += float(jf_loss)
        return jf_loss

    @classmethod
    def _batch_preprocessing(
            cls,
            obj_ref,
            features: torch.Tensor,
            labels: torch.Tensor,
            ) -> tuple:
        features.requires_grad_()
        return (features, labels)

    @classmethod
    def _epoch_preprocessing_train(cls, obj_ref) -> None:
        obj_ref.cumm_jf_loss = 0.0

    @classmethod
    def _epoch_postprocessing_train(cls, obj_ref) -> None:
        obj_ref.log["JFReg_loss"].append(obj_ref.cumm_jf_loss)

    @classmethod
    def reg_eval(
            cls,
            obj_ref,
            features: torch.Tensor,
            labels: torch.Tensor,
            predictions: torch.Tensor,
            hidden_repr: torch.Tensor,
            ) -> None:
        jf_score = cls._JF(features, labels, predictions, hidden_repr)
        obj_ref.eval_JFReg_score += float(jf_score)

    @classmethod
    def _epoch_preprocessing_eval(cls, obj_ref) -> None:
        obj_ref.eval_JFReg_score = 0.0

    @classmethod
    def _epoch_postprocessing_eval(cls, obj_ref) -> None:
        obj_ref.log["eval_JFReg_score"].append(obj_ref.eval_JFReg_score)

