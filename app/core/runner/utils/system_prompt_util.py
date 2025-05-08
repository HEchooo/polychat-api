

class SystemPromptUtils:
    SPLITTER = "--|--|--"

    @staticmethod
    def merge(system1: str, system2: str) -> str:
        return f"{system1.strip()}\n{SystemPromptUtils.SPLITTER}\n{system2.strip()}"

    @staticmethod
    def split(merged: str) -> tuple[str, str]:
        parts = merged.split(SystemPromptUtils.SPLITTER, maxsplit=1)
        system1 = parts[0].strip() if len(parts) > 0 else ""
        system2 = parts[1].strip() if len(parts) > 1 else ""
        return system1, system2

    @staticmethod
    def is_merged(text: str) -> bool:
        return SystemPromptUtils.SPLITTER in text
