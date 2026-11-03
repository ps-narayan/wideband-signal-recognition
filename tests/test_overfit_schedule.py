from wideband_recognition.yolo_schedule import expected_optimizer_schedule


def test_pinned_default_batch_accumulation_is_counted_as_optimizer_updates():
    schedule = expected_optimizer_schedule(
        epochs=100,
        batches_per_epoch=2,
        batch_size=4,
        nominal_batch_size=64,
        warmup_epochs=3,
    )

    assert schedule["micro_batches_total"] == 200
    assert schedule["steady_state_accumulation"] == 16
    assert schedule["optimizer_updates_expected_from_pinned_loop"] == 25
    assert schedule["optimizer_updates_during_warmup"] == 18


def test_small_set_diagnostic_can_update_each_batch():
    schedule = expected_optimizer_schedule(
        epochs=100,
        batches_per_epoch=2,
        batch_size=4,
        nominal_batch_size=4,
        warmup_epochs=3,
    )

    assert schedule["steady_state_accumulation"] == 1
    assert schedule["optimizer_updates_expected_from_pinned_loop"] == 200


def test_pinned_default_accumulation_starves_medium_cpu_profile():
    schedule = expected_optimizer_schedule(
        epochs=15,
        batches_per_epoch=6,
        batch_size=2,
        nominal_batch_size=64,
        warmup_epochs=3,
    )

    assert schedule["micro_batches_total"] == 90
    assert schedule["optimizer_updates_expected_from_pinned_loop"] == 10
    assert schedule["optimizer_updates_during_warmup"] == 10


def test_medium_cpu_batch_aligned_schedule_leaves_warmup_and_updates_all_batches():
    schedule = expected_optimizer_schedule(
        epochs=15,
        batches_per_epoch=6,
        batch_size=2,
        nominal_batch_size=2,
        warmup_epochs=3,
        warmup_min_iterations=10,
    )

    assert schedule["micro_batches_total"] == 90
    assert schedule["optimizer_updates_expected_from_pinned_loop"] == 90
    assert schedule["optimizer_updates_during_warmup"] == 19
