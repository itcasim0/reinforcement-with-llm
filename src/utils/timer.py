import time


class Timer:
    def __init__(self):
        """Initialize Timer and start timing."""
        self.start()

    def clear_sections(self):
        """Clear all recorded time sections."""
        self._sections: list[tuple[float, str]] = []

    def start(self):
        """Start (or restart) the timer."""
        self._start = time.perf_counter()
        self.clear_sections()

    def mark(self, name: str = "section"):
        """Mark a named section in time.

        Args:
            name: A label for this time section.
        """
        current_time = time.perf_counter()
        self._sections.append((current_time, name))

        return self

    def _calc_time_diff(self, cur: float, prev: float) -> float:
        """Calculate time difference between two timestamps."""
        return cur - prev

    def get_section_time(self, index: int) -> tuple[float, str]:
        """Get the time from start to a specific section.

        Args:
            index: Index of the section.

        Returns:
            (elapsed_time_from_start, section_name)
        """
        if not (0 <= index < len(self._sections)):
            raise IndexError("Invalid section index.")
        section_time, name = self._sections[index]
        return self._calc_time_diff(section_time, self._start), name

    def get_interval_duration(self, index: int) -> tuple[float, str]:
        """Get the duration between this section and the previous one.

        Args:
            index: Index of the section.

        Returns:
            (interval_duration, section_name)
        """
        if not (0 <= index < len(self._sections)):
            raise IndexError("Invalid section index.")
        current_time, name = self._sections[index]
        prev_time = self._start if index == 0 else self._sections[index - 1][0]
        return self._calc_time_diff(current_time, prev_time), name

    def get_total_duration(self) -> float:
        """Get total time from start to last section."""
        if not self._sections:
            raise RuntimeError("No sections marked.")
        return self._calc_time_diff(self._sections[-1][0], self._start)

    def get_summary(self) -> str:
        """Get a formatted summary of all sections and total time."""
        if not self._sections:
            return "No sections marked."

        lines = []
        max_len = 0
        for idx, _ in enumerate(self._sections):
            interval_time, name = self.get_interval_duration(idx)
            line = f"[{idx:3d}] {interval_time:.4f} sec <-- {name}"
            max_len = max(max_len, len(line))
            lines.append(line)

        line_sep = "-" * max_len
        total_time = self.get_total_duration()
        summary = "\n".join(lines)
        return f"{line_sep}\n{summary}\n{line_sep}\nTotal {total_time:.3f} sec"
