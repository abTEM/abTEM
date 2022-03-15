from hypothesis import settings

settings.register_profile("dev", max_examples=10, deadline=None)
settings.load_profile("dev")