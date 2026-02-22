from dataclasses import dataclass

@dataclass
class LevelOfAutomation:
    """Level of Automation"""
    level: int
    low_threshold: float
    high_threshold: float

@dataclass
class ClassificationProbabilities:
    """Classification probabilities"""
    ptp: float
    pfp: float
    pfn: float
    ptn: float


@dataclass
class HumanConditional:
    alarm: ClassificationProbabilities
    no_alarm: ClassificationProbabilities

@dataclass
class HumanConditional2DSS:
    aa: ClassificationProbabilities      # DSS1=A, DSS2=A
    an: ClassificationProbabilities      # DSS1=A, DSS2=NA
    na: ClassificationProbabilities      # DSS1=NA, DSS2=A
    nn: ClassificationProbabilities      # DSS1=NA, DSS2=NA


@dataclass
class Payoffs:
    """Classification probabilities"""
    V_TP: float
    V_FP: float
    V_FN: float
    V_TN: float

@dataclass
class OutcomeVariables:
    """Workload, Accuracy"""
    workload: float
    accuracy: float

