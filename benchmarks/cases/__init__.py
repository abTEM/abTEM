"""Benchmark case definitions, registered with ``abtem_bench.registry.case``.

Rules (enforced where possible by the registry): explicit ``gpts`` (never
``sampling=``), explicit ``max_batch`` and chunk size except in ``auto``
variants, all three tiers declared, setup outside ``run()``, only ``run()``
timed. Import abtem inside the case function so the registry loads without
importing abtem.
"""
