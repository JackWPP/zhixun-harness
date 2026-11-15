"""Evidence-driven, train-only CPU benchmark optimization harness with RuleDrivenHarnessTrainer."""
from .training import (
    BudgetExhausted,
    CostLedger,
    DiagnosticProbe,
    DiagnosticState,
    ExperienceBuffer,
    RuleDrivenHarnessTrainer,
    TaskSpec,
    TrajectoryRecord,
    get_default_task_specs,
    run_candidate,
    train_epoch,
)
from .strategies import (
    STRATEGY_IDS,
    Strategy,
    StrategyRecommenderMLP,
    choose_strategies,
    diagnosis_to_state_vector,
    rule_select_candidates,
)

__version__ = "0.2.0"
__all__ = [
    "BudgetExhausted",
    "CostLedger",
    "DiagnosticProbe",
    "DiagnosticState",
    "ExperienceBuffer",
    "RuleDrivenHarnessTrainer",
    "STRATEGY_IDS",
    "Strategy",
    "StrategyRecommenderMLP",
    "TaskSpec",
    "TrajectoryRecord",
    "choose_strategies",
    "diagnosis_to_state_vector",
    "get_default_task_specs",
    "rule_select_candidates",
    "run_candidate",
    "train_epoch",
]
