"""Print the version behind the alias champion. The deploy smoke test compares it with /health."""

from lucerne_temp.common.registry import FileModelStore

print(FileModelStore().version_of("champion"))
