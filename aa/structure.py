from typing import Dict, List, Any


class Structure:
    def __init__(self, name: str, sequence: str) -> None:
        self.name = name
        self.sequence = sequence

    def get_descriptors(self, desc_names: List[str]) -> Dict[str, int]:
        pass
