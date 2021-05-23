from __future__ import annotations


class Point:
    row: int
    column: int

    def __init__(self, row: int = -1, column: int = -1):
        self.row = row
        self.column = column

    def is_null(self) -> bool:
        return self.row == self.column == -1

    def get_coords(self) -> tuple[int, int]:
        return self.row, self.column

    def set_coords(self, row: int, column: int) -> None:
        self.row = row
        self.column = column

    def equals(self, row: int, column: int) -> bool:
        return self.row == row and self.column == column

    def reset(self) -> None:
        self.row, self.column = -1, -1

    def __eq__(self, other: Point) -> bool:
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return int(self.row * 100 + self.column * 10)

    def __repr__(self):
        return "Point{row: %s, column: %s}" % (self.row, self.column)
