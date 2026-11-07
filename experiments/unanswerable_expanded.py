if __package__:
    from .unanswerable_main import analyze as analyze_probe
    from .unanswerable_main import (
        command,
        detection_metrics as main_metrics,
        run_probe as run_main,
    )
else:
    from unanswerable_main import analyze as analyze_probe
    from unanswerable_main import (
        command,
        detection_metrics as main_metrics,
        run_probe as run_main,
    )


def analyze(plans):
    return analyze_probe(plans, listed=True)


def detection_metrics(plan, problem_ids, unanswerable_ids):
    return main_metrics(plan, problem_ids, unanswerable_ids, listed=True)


def run_probe(
    problems,
    references,
    configuration,
    settings,
    ratios=(0.25, 0.5, 0.75, 1.0),
    seed=42,
    pool_size=10,
    run_id=0,
    output=None,
):
    return run_main(
        problems,
        references,
        configuration,
        settings,
        ratios,
        seed,
        pool_size,
        run_id,
        listed=True,
        output=output,
    )


if __name__ == "__main__":
    command(listed=True)
