from dataclasses import dataclass
import numpy as np
from math import erf, sqrt

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


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


def compute_threshold(Ps: float, payoffs: Payoffs) -> float:
    """Compute optimal threshold

    :param Ps: prior probability to signal
    :param payoffs: payoff matrix from the 4 possible classifications
    :return: optimal threshold for one detector
    """
    return (float(1-Ps)/float(Ps)) * (float(payoffs.V_FP - payoffs.V_TN)/float(payoffs.V_FN - payoffs.V_TP))


# Compute 1 detector classification probabilities based on sensitivity and threshold
def compute_probs(sensitivity: float, threshold: float) -> ClassificationProbabilities:
    """Compute classification probabilities

    :param sensitivity: detector sensitivity
    :param threshold: detector single decision threshold
    :return: detector's classification probabilities
    """

    z_tp = float((sensitivity**2 - 2*np.log(threshold))/(2*sensitivity))
    z_fp = -1 * float((sensitivity**2 + 2*np.log(threshold))/(2*sensitivity))
    ptp = norm_cdf(z_tp)
    pfp = norm_cdf(z_fp)
    pfn = 1 - ptp
    ptn = 1- pfp
    return ClassificationProbabilities(ptp=ptp, pfp=pfp, pfn=pfn, ptn=ptn)


def compute_conditional_probs(
        Ps: float,
        sensitivity_human: float,
        sensitivity_system: float,
        threshold_system: float,
        payoffs: Payoffs
) -> HumanConditional:
    Pn = 1 - Ps
    system_classification_probs = compute_probs(sensitivity_system, threshold_system)

    # Compute PPV and NPV correctly
    PPV = (Ps * system_classification_probs.ptp) / (
            Ps * system_classification_probs.ptp + Pn * system_classification_probs.pfp
    )

    NPV = (Pn * system_classification_probs.ptn) / (
            Pn * system_classification_probs.ptn + Ps * system_classification_probs.pfn
    )

    # Compute optimal thresholds for alarm and no alarm events
    threshold_a = compute_threshold(PPV, payoffs)        # For alarm events
    threshold_na = compute_threshold(1 - NPV, payoffs)   # For no alarm events

    # Compute human conditional probabilities
    conditional_prpbs = HumanConditional(
        alarm=compute_probs(sensitivity_human, threshold_a),
        no_alarm=compute_probs(sensitivity_human, threshold_na)
    )

    return conditional_prpbs

def combined_sensitivity(source_1: float, source_2: float) -> float:
    """Compute the combined sensitivity based on the two independent sources

    :param source_1: source 1 sensitivity
    :param source_2: source 2 sensitivity
    :return combined sensitivity
    """
    return np.sqrt(source_1**2 + source_2**2)


def ev_one_detector(Ps: float, classification_probs: ClassificationProbabilities, payoffs: Payoffs) -> float:
    """Compute Expected value of 1 detector only

    :param Ps: prior probability to signal
    :param classification_probs: detector classification probabilities
    :param payoffs: payoff matrix
    :return: Expected Value
    """
    return (Ps * (classification_probs.ptp * payoffs.V_TP + classification_probs.pfn * payoffs.V_FN) +
            (1-Ps)*(classification_probs.pfp * payoffs.V_FP + classification_probs.ptn * payoffs.V_TN))


def ev_human_one_dss(
        Ps: float,
        DSS: ClassificationProbabilities,
        human: HumanConditional,
        payoffs: Payoffs,
        c: float = 0
) -> float:
    """
    Compute expected value for a human interacting with one DSS.
    """
    # Signal present branch
    ev_signal = (DSS.ptp * (human.alarm.ptp * payoffs.V_TP + human.alarm.pfn * payoffs.V_FN) +
                 DSS.pfn * (human.no_alarm.ptp * payoffs.V_TP + human.no_alarm.pfn * payoffs.V_FN))

    # Noise present branch
    ev_noise = (DSS.pfp * (human.alarm.pfp * payoffs.V_FP + human.alarm.ptn * payoffs.V_TN) +
                DSS.ptn * (human.no_alarm.pfp * payoffs.V_FP + human.no_alarm.ptn * payoffs.V_TN))

    # Total expected value
    return Ps * ev_signal + (1 - Ps) * ev_noise - c


def ev_human_two_dss(
        Ps: float,
        DSS1: ClassificationProbabilities,
        DSS2: ClassificationProbabilities,
        human_sensitivity: float,
        payoffs: Payoffs
) -> float:
    """
    Compute expected value for a human interacting with two independent DSS systems.
    Human have 4 different thresholds depending on DSS outcomes: (A,A), (NA,NA), (A,NA), (NA,A)

    :param Ps: prior probability of signal
    :param DSS1: classification probabilities of DSS1
    :param DSS2: classification probabilities of DSS2
    :param human_sensitivity: sensitivity of the human
    :param payoffs: payoff matrix
    :return: total expected value
    """
    Pn = 1 - Ps
    outcomes = [('A', 'A'), ('NA', 'NA'), ('A', 'NA'), ('NA', 'A')]
    ev_total = 0.0

    for d1, d2 in outcomes:
        # 1️⃣ Compute DSS conditional probabilities for this outcome
        if d1 == 'A':
            p1_s = DSS1.ptp
            p1_n = DSS1.pfp
        else:  # 'NA'
            p1_s = DSS1.pfn
            p1_n = DSS1.ptn

        if d2 == 'A':
            p2_s = DSS2.ptp
            p2_n = DSS2.pfp
        else:  # 'NA'
            p2_s = DSS2.pfn
            p2_n = DSS2.ptn

        # 2️⃣ Joint probability of this DSS outcome
        P_dss_given_S = p1_s * p2_s
        P_dss_given_N = p1_n * p2_n
        P_outcome = Ps * P_dss_given_S + Pn * P_dss_given_N

        # 3️⃣ Posterior probabilities
        P_S_given = (Ps * P_dss_given_S) / P_outcome
        P_N_given = 1 - P_S_given

        # 4️⃣ Compute human threshold for this outcome
        threshold = compute_threshold(P_S_given, payoffs)
        human_conf = compute_probs(human_sensitivity, threshold)

        # 5️⃣ EV for this outcome
        ev_outcome = (
                P_S_given * (human_conf.ptp * payoffs.V_TP + human_conf.pfn * payoffs.V_FN) +
                P_N_given * (human_conf.pfp * payoffs.V_FP + human_conf.ptn * payoffs.V_TN)
        )

        # 6️⃣ Weighted EV by probability of DSS outcome
        ev_total += P_outcome * ev_outcome

    return ev_total

def compute_all_ev(
        Ps: float,
        source_1_sensitivity: float,
        source_2_sensitivity: float,
        DSS1_sensitivity: float = None,
        DSS1_threshold: float = None,
        DSS2_sensitivity: float = None,
        DSS2_threshold: float = None,
        payoffs: Payoffs = None,
        DSS1_cost: float = 0.0,
        DSS2_cost: float = 0.0
) -> dict:
    """
    Compute expected value for human alone, human+one DSS, and human+two DSS.
    If DSS thresholds are None, compute optimal thresholds automatically.

    :param Ps: prior probability of signal
    :param source_1_sensitivity: source 1 sensitivity (d')
    :param source_2_sensitivity: source 1 sensitivity (d')
    :param DSS1_sensitivity: DSS1 sensitivity (d'), optional
    :param DSS1_threshold: DSS1 decision threshold, optional
    :param DSS2_sensitivity: DSS2 sensitivity (d'), optional
    :param DSS2_threshold: DSS2 decision threshold, optional
    :param payoffs: payoff matrix
    :param DSS1_cost: cost of using DSS1
    :param DSS2_cost: cost of using DSS2
    :return: dict with keys 'human_alone', 'human_one_dss', 'human_two_dss'
    """
    results = {}

    human_sensitivity = combined_sensitivity(source_1_sensitivity, source_2_sensitivity)

    # 1️⃣ Human alone (no DSS)
    threshold_human_alone = compute_threshold(Ps, payoffs)
    human_alone_conf = compute_probs(human_sensitivity, threshold_human_alone)
    ev_human_alone = ev_one_detector(Ps, human_alone_conf, payoffs)
    results['human_alone'] = float(round(ev_human_alone, 2))

    # 2️⃣ Human + one DSS (first)
    # Compute optimal threshold if not provided
    if DSS1_threshold is None:
        DSS1_threshold = compute_threshold(Ps, payoffs)
    DSS1_conf = compute_probs(DSS1_sensitivity, DSS1_threshold)
    human_cond_one_dss = compute_conditional_probs(Ps, human_sensitivity,
                                                   DSS1_sensitivity,
                                                   DSS1_threshold, payoffs)
    ev_one = ev_human_one_dss(Ps, DSS1_conf, human_cond_one_dss, payoffs, c=DSS1_cost)
    results['human_first_dss'] = float(round(ev_one, 2))

    # 2️⃣ Human + one DSS (second)
    # Compute optimal threshold if not provided
    if DSS2_threshold is None:
        DSS2_threshold = compute_threshold(Ps, payoffs)
    DSS2_conf = compute_probs(DSS2_sensitivity, DSS2_threshold)
    human_cond_one_dss = compute_conditional_probs(Ps, human_sensitivity,
                                                   DSS2_sensitivity,
                                                   DSS2_threshold, payoffs)
    ev_one = ev_human_one_dss(Ps, DSS2_conf, human_cond_one_dss, payoffs, c=DSS2_cost)
    results['human_second_dss'] = float(round(ev_one, 2))

    # 3️⃣ Human + two DSS
    # Compute optimal thresholds if not provided
    if DSS1_threshold is None:
        DSS1_threshold = compute_threshold(Ps, payoffs)
    if DSS2_threshold is None:
        DSS2_threshold = compute_threshold(Ps, payoffs)

    DSS1_conf = compute_probs(DSS1_sensitivity, DSS1_threshold)
    DSS2_conf = compute_probs(DSS2_sensitivity, DSS2_threshold)
    ev_two = ev_human_two_dss(Ps, DSS1_conf, DSS2_conf, human_sensitivity, payoffs)
    # Subtract DSS costs
    ev_two -= DSS1_cost + DSS2_cost
    results['human_two_dss'] = float(round(ev_two, 2))

    return results

# streamlit run ui/app.py