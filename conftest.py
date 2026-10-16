# Empty on purpose. Its presence makes pytest insert the repo root onto
# sys.path during collection, so test modules under tests/ (which has its
# own __init__.py chain) can `import scripts....` regardless of cwd.
