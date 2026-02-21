from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    files: Sequence[str]
    has_header: bool
    label_cols: Sequence[str]
    multiclass_col: str | None
    binary_col: str | None
    possible_timestamp_cols: Sequence[str]
    possible_entity_cols: Sequence[str]

    def resolved_files(self, data_dir: Path) -> list[Path]:
        return [data_dir / rel_path for rel_path in self.files]


KDD_FEATURE_COLUMNS: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty_level",
]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "kdd": DatasetSpec(
        name="kdd",
        files=["kdd/KDDTrain+.txt", "kdd/KDDTest+.txt"],
        has_header=False,
        label_cols=["label", "difficulty_level"],
        multiclass_col="label",
        binary_col=None,
        possible_timestamp_cols=["timestamp", "ts", "stime", "ltime"],
        possible_entity_cols=[
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "sport",
            "dport",
            "state",
            "proto",
        ],
    ),
    "unsw-nb15": DatasetSpec(
        name="unsw-nb15",
        files=[
            "unsw-nb15/UNSW_NB15_training-set.csv",
            "unsw-nb15/UNSW_NB15_testing-set.csv",
        ],
        has_header=True,
        label_cols=["label", "attack_cat"],
        multiclass_col="attack_cat",
        binary_col="label",
        possible_timestamp_cols=["timestamp", "ts", "stime", "ltime"],
        possible_entity_cols=[
            "srcip",
            "dstip",
            "sport",
            "dport",
            "proto",
            "service",
            "state",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
        ],
    ),
    "ton-iot": DatasetSpec(
        name="ton-iot",
        files=["ton-iot/train_test_network.csv"],
        has_header=True,
        label_cols=["label", "type"],
        multiclass_col="type",
        binary_col="label",
        possible_timestamp_cols=["timestamp", "ts", "stime", "ltime"],
        possible_entity_cols=[
            "src_ip",
            "dst_ip",
            "srcip",
            "dstip",
            "src_port",
            "dst_port",
            "sport",
            "dport",
            "proto",
            "service",
            "state",
        ],
    ),
}
