"""Pure-Python lifetime boundary for streamed expert compute batches.

Keeping this helper independent of MLX makes the subtle generator ownership rule
unit-testable: the previous yielded mapping must be deleted before ``next()`` is
called, otherwise Python's loop target retains it while the next fetch runs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


def consume_expert_batches(
    batches: Iterable[tuple[list[int], dict]],
    consume: Callable[[list[int], dict], None],
) -> None:
    """Consume and release batch N before asking the producer for batch N+1.

    If compute raises, close a generator producer explicitly.  That runs any
    producer-side ``finally`` block immediately instead of relying on delayed
    garbage collection to release a yielded page mapping or file handle.
    """
    batch_iter = iter(batches)
    try:
        while True:
            try:
                batch_ids, experts = next(batch_iter)
            except StopIteration:
                return
            try:
                consume(batch_ids, experts)
            finally:
                # Do not rewrite this as ``for batch_ids, experts in batches``.
                # A for-loop calls next() before rebinding the targets, so two
                # large mappings coexist.  The finally also releases the current
                # payload before an exception escapes.
                del experts, batch_ids
    finally:
        close = getattr(batch_iter, "close", None)
        if close is not None:
            close()
