"""Official V2 aggregation must never silently lose missing/error coverage."""

import pytest

from tau2.data_model.benchmark import (
    BenchmarkFrame,
    CellFrame,
    TaskTrial,
    TrialStatus,
)
from tau2.metrics.benchmark import summarize


@pytest.fixture
def frame():
    return BenchmarkFrame(
        benchmark_version="voice-v2",
        revision="fixture-v1",
        mode="voice",
        trials=[0, 1],
        cells=[
            CellFrame(
                language=language,
                domain=domain,
                canonical_task_ids=[str(i) for i in range(50)],
                binding_revision=f"{language}-{domain}-v1",
            )
            for language in ("en", "es", "pt", "hi", "zh", "ko")
            for domain in ("airline", "retail", "telecom")
        ],
    )


def result(cell, task="0", trial=0, reward=1.0, status=TrialStatus.SCORED):
    return TaskTrial(
        language=cell.language,
        domain=cell.domain,
        canonical_task_id=task,
        trial=trial,
        condition_fingerprint="a" * 64,
        binding_revision=cell.binding_revision,
        status=status,
        reward=reward,
    )


def test_complete_matrix_averages_trials_tasks_and_cells(frame):
    rows = [
        result(cell, task, trial, reward=float(i == 0))
        for i, cell in enumerate(frame.cells)
        for task in cell.canonical_task_ids
        for trial in frame.trials
    ]
    report = summarize(frame, rows)
    assert report.complete
    assert report.score == pytest.approx(1 / 18)
    assert report.by_language["en"].score == pytest.approx(1 / 3)
    assert report.by_domain["airline"].score == pytest.approx(1 / 6)
    assert report.by_language["es"].score == 0
    assert report.expected_trials == report.scored_trials == 1800
    assert report.missing_trials == report.error_trials == 0


def test_partial_observed_score_does_not_pool_unequal_trial_counts(frame):
    rows = [
        result(frame.cells[0], "0", 0, 0),
        result(frame.cells[0], "0", 1, 0),
        result(frame.cells[0], "1", 0, 1),
        result(frame.cells[1], "0", 0, 1),
        result(frame.cells[2], "0", 0, None, TrialStatus.ERROR),
    ]
    report = summarize(frame, rows)
    assert report.score is None
    assert not report.complete
    assert report.observed_score == pytest.approx(0.75)
    assert report.by_language["en"].score is None
    assert report.by_domain["airline"].score is None
    assert report.cells[0].observed_score == 0.5
    assert report.error_trials == 1
    assert report.missing_trials == 1795
    assert len(report.cells) == 18
    assert report.cells[-1].observed_score is None


@pytest.mark.parametrize(
    "change", ["condition", "binding", "task", "trial", "duplicate"]
)
def test_incompatible_or_ambiguous_results_are_rejected(frame, change):
    row = result(frame.cells[0])
    other = result(frame.cells[0], "1")
    if change == "condition":
        other.condition_fingerprint = "b" * 64
    elif change == "binding":
        other.binding_revision = "different-pack"
    elif change == "task":
        other.canonical_task_id = "unknown"
    elif change == "trial":
        other.trial = 10
    else:
        other = row
    with pytest.raises(ValueError):
        summarize(frame, [row, other])


def test_error_cannot_carry_reward_and_unscored_remains_incomplete(frame):
    with pytest.raises(ValueError):
        result(frame.cells[0], reward=1, status=TrialStatus.ERROR)
    report = summarize(
        frame, [result(frame.cells[0], reward=None, status=TrialStatus.UNSCORED)]
    )
    assert report.unscored_trials == 1
    assert report.score is None


def test_frame_rejects_duplicate_cells_and_inconsistent_domains(frame):
    with pytest.raises(ValueError, match="unique"):
        BenchmarkFrame(
            benchmark_version="voice-v2",
            revision="bad",
            mode="voice",
            trials=[0],
            cells=frame.cells + [frame.cells[0]],
        )
    with pytest.raises(ValueError, match="every released domain"):
        BenchmarkFrame(
            benchmark_version="voice-v2",
            revision="bad",
            mode="voice",
            trials=[0],
            cells=frame.cells[:-1],
        )


def test_official_run_requires_a_separately_pinned_semantic_judge(frame):
    from tau2.data_model.benchmark import BenchmarkRunInfo, encode_conditions
    from tau2.data_model.run_config import AudioNativeConfig
    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        SemanticCommunicateSettings,
    )
    from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig

    config = TextRunConfig(
        communicate_scoring=CommunicateScoring(scorer="substring"),
        benchmark=BenchmarkRunInfo(
            benchmark_version="voice-v2",
            frame_revision=frame.revision,
            language="en",
            domain="airline",
            binding_revision="fixture",
            canonical_task_ids=frame.cells[0].canonical_task_ids,
            selected_task_ids=frame.cells[0].canonical_task_ids,
            trials=[0],
            condition_fingerprint="a" * 64,
            conditions=encode_conditions(
                VoiceRunConfig(language="en", audio_native_config=AudioNativeConfig())
            ),
        ),
    )
    with pytest.raises(ValueError, match="pinned semantic"):
        config.validate()
    config.communicate_scoring = CommunicateScoring(
        judge=SemanticCommunicateSettings(model="explicit-test-judge", model_args={})
    )
    config.validate()
