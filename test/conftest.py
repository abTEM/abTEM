from hypothesis import settings

settings.register_profile("dev", max_examples=10, print_blob=True, deadline=None)
settings.load_profile("dev")