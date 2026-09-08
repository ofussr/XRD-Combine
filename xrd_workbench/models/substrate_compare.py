"""Toolkit-independent state for the substrate comparison workspace."""

from __future__ import annotations

from dataclasses import dataclass, field

from .scan import Scan1D


@dataclass(eq=False)
class ComparisonItem:
    """One draggable measurement or substrate card."""

    kind: str
    name: str
    scan: Scan1D
    x: int
    y: int
    home_x: int
    home_y: int
    width: int = 180
    height: int = 30
    group: int | None = None
    active: bool = False


@dataclass
class ComparisonAssembly:
    """An active group that can be plotted in the gallery or viewer."""

    group_id: int
    label: str
    items: list[ComparisonItem]
    view: dict[str, object] = field(default_factory=dict)


class ComparisonWorkspace:
    """Own comparison items, grouping and saved view settings without a GUI."""

    def __init__(self) -> None:
        self.items: list[ComparisonItem] = []
        self.groups: dict[int, set[ComparisonItem]] = {}
        self.group_active: dict[int, bool] = {}
        self.group_views: dict[int, dict[str, object]] = {}
        self.next_group_id = 1

    def unique_name(self, wanted: str) -> str:
        existing = {item.name for item in self.items}
        if wanted not in existing:
            return wanted
        index = 1
        while f"{wanted}_{index}" in existing:
            index += 1
        return f"{wanted}_{index}"

    def add_item(
        self,
        *,
        kind: str,
        name: str,
        scan: Scan1D,
        x: int,
        y: int,
        width: int = 180,
        height: int = 30,
    ) -> ComparisonItem:
        item = ComparisonItem(
            kind=kind,
            name=name,
            scan=scan,
            x=x,
            y=y,
            home_x=x,
            home_y=y,
            width=width,
            height=height,
        )
        self.items.append(item)
        return item

    def remove_kind(self, kind: str) -> list[ComparisonItem]:
        self.reset_layout()
        removed = [item for item in self.items if item.kind == kind]
        self.items[:] = [item for item in self.items if item.kind != kind]
        return removed

    def reset_layout(self) -> None:
        self.groups.clear()
        self.group_active.clear()
        self.group_views.clear()
        for item in self.items:
            item.group = None
            item.active = False
            item.x, item.y = item.home_x, item.home_y

    @staticmethod
    def overlap_area(first: ComparisonItem, second: ComparisonItem) -> int:
        ax1, ay1 = first.x, first.y
        ax2, ay2 = ax1 + first.width, ay1 + first.height
        bx1, by1 = second.x, second.y
        bx2, by2 = bx1 + second.width, by1 + second.height
        width = max(0, min(ax2, bx2) - max(ax1, bx1))
        height = max(0, min(ay2, by2) - max(ay1, by1))
        return width * height

    @staticmethod
    def move_item(item: ComparisonItem, x: int, y: int) -> None:
        item.x, item.y = x, y

    @staticmethod
    def set_item_active(item: ComparisonItem, active: bool) -> None:
        item.active = active

    def set_group_active(self, group_id: int, active: bool) -> None:
        self.group_active[group_id] = active
        for item in self.groups[group_id]:
            self.set_item_active(item, active)

    def layout_group(self, group_id: int, x: int, y: int) -> None:
        members = sorted(
            self.groups[group_id],
            key=lambda item: (0 if item.kind == "file" else 1, item.name),
        )
        for item in members:
            self.move_item(item, x, y)
            y += item.height

    def create_group(self, *members: ComparisonItem) -> int:
        if len(members) < 2:
            raise ValueError("A comparison group needs at least two items.")
        group_id = self.next_group_id
        self.next_group_id += 1
        self.groups[group_id] = set(members)
        self.group_active[group_id] = True
        self.group_views[group_id] = {}
        for item in members:
            item.group = group_id
            self.set_item_active(item, True)
        self.layout_group(group_id, members[0].x, members[0].y)
        return group_id

    def add_to_group(self, group_id: int, item: ComparisonItem) -> None:
        self.groups[group_id].add(item)
        item.group = group_id
        self.set_item_active(item, self.group_active[group_id])
        anchor = next(iter(self.groups[group_id]))
        self.layout_group(group_id, anchor.x, anchor.y)

    def merge_groups(self, first: int, second: int) -> int:
        if first == second:
            return first
        for item in list(self.groups[second]):
            item.group = first
            self.groups[first].add(item)
        self.group_active[first] = self.group_active[first] or self.group_active[second]
        self.group_views[first] = (
            self.group_views.get(first) or self.group_views.get(second) or {}
        )
        del self.groups[second]
        del self.group_active[second]
        self.group_views.pop(second, None)
        anchor = next(iter(self.groups[first]))
        self.layout_group(first, anchor.x, anchor.y)
        self.set_group_active(first, self.group_active[first])
        return first

    def remove_from_group(self, item: ComparisonItem) -> None:
        group_id = item.group
        if group_id is None:
            return
        self.groups[group_id].discard(item)
        item.group = None
        self.set_item_active(item, False)
        if len(self.groups[group_id]) < 2:
            if self.groups[group_id]:
                remaining = next(iter(self.groups[group_id]))
                remaining.group = None
                self.set_item_active(remaining, False)
            del self.groups[group_id]
            del self.group_active[group_id]
            self.group_views.pop(group_id, None)
            return
        anchor = next(iter(self.groups[group_id]))
        self.layout_group(group_id, anchor.x, anchor.y)

    def active_assemblies(self) -> list[ComparisonAssembly]:
        result: list[ComparisonAssembly] = []
        for group_id, members in self.groups.items():
            if not self.group_active[group_id]:
                continue
            ordered = sorted(
                members,
                key=lambda item: (0 if item.kind == "file" else 1, item.name),
            )
            result.append(
                ComparisonAssembly(
                    group_id=group_id,
                    label=" + ".join(item.name for item in ordered),
                    items=ordered,
                    view=dict(self.group_views.get(group_id, {})),
                )
            )
        return result
