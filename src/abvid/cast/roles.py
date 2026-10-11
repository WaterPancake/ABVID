"""Metadata-only role checks for the frozen CAST-4A grouped design.

These checks admit ancestry to one selected training role. They do not validate
fit numerics, approve cache reuse, open audio, or execute a scientific operation.
Population statistics must be rebuilt for the current fold; only same-parent
fits can be considered by a separate provenance and numerical reuse audit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType

from abvid.cast import grouped_design
from abvid.paths import resolve
from abvid.protocol import load
from abvid.provenance.hashing import canonical


ALIASES = {"class": "canonical_class", "class_label": "canonical_class",
           "group": "provenance_group_id", "input_id": "file_id", "parent_id": "file_id"}


def _sequence(values, name):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an explicit sequence")
    return list(values)


@dataclass(frozen=True)
class FoldRole:
    """Immutable allowlist and metadata authority for one verified partition."""

    partition_id: str
    analysis: str
    train_ids: tuple[str, ...]
    held_ids: tuple[str, ...]
    held_groups: tuple[str, ...]
    training_groups: tuple[str, ...]
    _source: Mapping[str, bytes] = field(repr=False)
    _held_links: Mapping[str, frozenset] = field(repr=False)
    _class_ids: Mapping[str, tuple[str, ...]] = field(repr=False)
    design_identity_sha256: str | None = None

    def source_metadata(self, file_id):
        """Return an independent copy; unknown and absent fields remain intact."""
        if not isinstance(file_id, str) or file_id not in self._source:
            raise ValueError("Unknown source parent")
        return json.loads(self._source[file_id])

    def class_train_ids(self, class_name):
        if class_name not in self._class_ids:
            raise ValueError("Only car and truck class roles are declared")
        return self._class_ids[class_name]

    def _population(self, class_name):
        return self.train_ids if class_name is None else self.class_train_ids(class_name)

    def _entry(self, entry, *, class_name=None):
        if isinstance(entry, str):
            metadata = {"file_id": entry}
        elif isinstance(entry, Mapping):
            metadata = {}
            for key, value in entry.items():
                key = ALIASES.get(key, key)
                if key in metadata and canonical(metadata[key]) != canonical(value):
                    raise ValueError(f"Conflicting ancestry metadata: {key}")
                metadata[key] = value
        else:
            raise ValueError("Ancestry entries must be parent IDs or metadata mappings")
        for key in grouped_design.LINKS:
            value = metadata.get(key)
            if grouped_design.known(value):
                try:
                    forbidden = value in self._held_links[key]
                except TypeError as exc:
                    raise ValueError(f"Ancestry link must be scalar: {key}") from exc
                if forbidden:
                    raise ValueError(f"Held ancestor is forbidden: {key}")
        parent = metadata.get("file_id")
        if not isinstance(parent, str) or parent not in self.train_ids:
            raise ValueError("Parent is not in the current selected training role")
        expected = self.source_metadata(parent)
        for key, value in metadata.items():
            if key not in expected or canonical(value) != canonical(expected[key]):
                raise ValueError(f"Ancestry differs from frozen source metadata: {key}")
        if class_name is not None and parent not in self.class_train_ids(class_name):
            raise ValueError("Parent belongs to a different class role")
        return parent

    def _parents(self, entries, *, class_name=None):
        checked = [self._entry(entry, class_name=class_name)
                   for entry in _sequence(entries, "Parent ancestry")]
        if not checked or len(checked) != len(set(checked)):
            raise ValueError("Parent population must be nonempty and contain no duplicates")
        return checked

    def validate_parent(self, file_id, *, ancestry=()):
        """Check one reusable-fit candidate and every supplied same-parent link.

        The caller's separate cache audit must supply the complete transitive
        chain and verify its hashes. Cross-parent calibration cannot pass here.
        """
        parent = self._entry(file_id)
        ancestors = [self._entry(entry) for entry in _sequence(ancestry, "Fit ancestry")]
        if any(ancestor != parent for ancestor in ancestors):
            raise ValueError("Per-parent fit ancestry must contain only the same original parent")
        return {"partition": self.partition_id, "file_id": parent,
                "source_metadata": self.source_metadata(parent), "ancestry_ids": ancestors,
                "role_validation_only": True, "fit_reuse_approved": False}

    def _calibration(self, entries, class_name):
        parents = self._parents(entries, class_name=class_name)
        if set(parents) != set(self._population(class_name)):
            raise ValueError("Calibration must use the complete current selected class population")
        return parents

    @staticmethod
    def _no_population_reuse(population_statistics_reused):
        if population_statistics_reused is not False:
            raise ValueError("Historical population statistics cannot be reused")

    def validate_bank(self, parent_ids, *, class_name=None, calibration_ids=None,
                      population_statistics_reused=False):
        """Require the full selected 380-parent bank, or its 190-parent class.

        A supplied calibration population must be complete too. This does not
        approve old population statistics, even if their parent IDs coincide.
        """
        self._no_population_reuse(population_statistics_reused)
        parents = self._parents(parent_ids, class_name=class_name)
        if set(parents) != set(self._population(class_name)):
            raise ValueError("Bank must contain the exact current selected training population")
        calibration = None if calibration_ids is None else self._calibration(calibration_ids, class_name)
        return {"partition": self.partition_id, "class": class_name, "parent_ids": parents,
                "calibration_ids": calibration, "held_groups": list(self.held_groups),
                "role_validation_only": True, "population_statistics_reused": False}

    def validate_donors(self, parent_ids, *, class_name, calibration_ids,
                        population_statistics_reused=False):
        """Check direct donor subsets and all parents behind class statistics."""
        self._no_population_reuse(population_statistics_reused)
        self.class_train_ids(class_name)
        parents = self._parents(parent_ids, class_name=class_name)
        calibration = self._calibration(calibration_ids, class_name)
        return {"partition": self.partition_id, "class": class_name, "parent_ids": parents,
                "calibration_ids": calibration, "held_groups": list(self.held_groups),
                "role_validation_only": True, "population_statistics_reused": False}


def build_roles(rows, partitions, *, design_identity_sha256=None):
    """Validate metadata already supplied by a caller; no file access occurs.

    Production callers should use load_roles to bind these checks to the pinned
    canonical config and independently replayed external freeze.
    """
    rows = _sequence(rows, "Source inventory")
    partitions = _sequence(partitions, "Partition inventory")
    if any(not isinstance(row, dict) or row.get("dataset_id") != "IDMT" for row in rows):
        raise ValueError("Role inventory may contain only admitted IDMT source metadata")
    for row in rows:
        if (type(row.get("class_id")) is not int or
                any(not isinstance(row.get(key), str) or not grouped_design.known(row[key])
                    for key in ("file_id", "provenance_group_id", "site_id"))):
            raise ValueError("Source role identifiers and class must be known")
        if any(grouped_design.known(row.get(key)) and not isinstance(row[key], str)
               for key in grouped_design.LINKS):
            raise ValueError("Known source ancestry links must be scalar strings")
    source = grouped_design.source_rows(json.loads(canonical(rows)))
    groups = {row["provenance_group_id"] for row in source}
    sites = {row["site_id"] for row in source}
    group_sites = {group: {row["site_id"] for row in source if row["provenance_group_id"] == group}
                   for group in groups}
    if len(groups) != 6 or len(sites) != 3 or any(len(value) != 1 for value in group_sites.values()):
        raise ValueError("Expected six source groups nested in three sites")
    if any(len({row["provenance_group_id"] for row in source if row["site_id"] == site}) != 2 for site in sites):
        raise ValueError("Each source site must retain both complete date groups")
    expected_ids = {f"group_{i}" for i in range(6)} | {f"site_{i}" for i in range(3)}
    if (any(not isinstance(part, dict) for part in partitions) or
            len(partitions) != 9 or {part.get("id") for part in partitions} != expected_ids):
        raise ValueError("All six group and three site partitions are required exactly once")
    source_bytes = MappingProxyType({row["file_id"]: canonical(row) for row in source})
    by_id = {row["file_id"]: row for row in source}
    result = {}
    labels = {"group": set(), "site": set()}
    for part in sorted(partitions, key=lambda value: value["id"]):
        if not {"analysis", "fold", "held_label", "held_groups", "train_ids", "held_ids"} <= set(part):
            raise ValueError("Incomplete partition metadata")
        analysis, fold, held_label = part["analysis"], part["fold"], part["held_label"]
        if analysis not in labels or type(fold) is not int or part["id"] != f"{analysis}_{fold}":
            raise ValueError("Invalid partition identity")
        expected_groups = ({held_label} if analysis == "group" else
                           {row["provenance_group_id"] for row in source if row["site_id"] == held_label})
        if (held_label not in (groups if analysis == "group" else sites) or
                part["held_groups"] != sorted(expected_groups)):
            raise ValueError("Partition must exclude the complete declared group or site")
        reconstructed = grouped_design.partition(
            source, part["train_ids"], part["held_ids"], analysis=analysis, fold=fold,
            held_label=held_label, held_groups=expected_groups)
        if canonical(reconstructed) != canonical(part):
            raise ValueError("Partition differs from its full metadata reconstruction")
        labels[analysis].add(held_label)
        held = [by_id[parent] for parent in part["held_ids"]]
        held_links = MappingProxyType({key: frozenset(row[key] for row in held if grouped_design.known(row.get(key)))
                                       for key in grouped_design.LINKS})
        class_ids = MappingProxyType({label: tuple(parent for parent in part["train_ids"]
                                                 if by_id[parent]["canonical_class"] == label)
                                      for label in ("car", "truck")})
        role = FoldRole(part["id"], analysis, tuple(part["train_ids"]), tuple(part["held_ids"]),
                        tuple(part["held_groups"]), tuple(sorted(part["training_group_counts"])),
                        source_bytes, held_links, class_ids, design_identity_sha256)
        result[role.partition_id] = role
    if labels["group"] != groups or labels["site"] != sites:
        raise ValueError("Partitions do not cover every source group and site")
    return result


def load_roles(config_reference=grouped_design.CONFIG):
    """Read only metadata and return roles bound to the verified design freeze."""
    config_path = resolve(grouped_design.CONFIG)
    if str(config_reference) != grouped_design.CONFIG and Path(config_reference).resolve() != config_path:
        raise ValueError("Role adapter requires the exact canonical grouped-design config")
    cfg = load(config_path, check_files=True)
    if cfg["id"] != "cast_grouped_coverage" or cfg["execution"]["mode"] != "metadata_only":
        raise ValueError("Role adapter requires the canonical metadata-only grouped design")
    if Path(grouped_design.__file__).resolve() != resolve(cfg["execution"]["script"]):
        raise ValueError("Grouped-design module differs from the hashed repository entry point")
    frozen_design = json.loads(resolve(cfg["contract"]["path"]).read_text())
    expected = grouped_design.payloads(frozen_design, config_path)
    record = grouped_design.verify(grouped_design.output_path(frozen_design), expected)
    rows = [json.loads(line) for line in expected["sources.jsonl"].splitlines()]
    partitions = json.loads(expected["folds.json"])
    return build_roles(rows, partitions, design_identity_sha256=record["payload_identity_sha256"])
